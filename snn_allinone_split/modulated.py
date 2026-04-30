"""Modulated SNN checkpoints, forward pass, analysis, and training."""
from .modulation import *

def _collect_stockpile_checkpoints(stockpile_dir: Union[str, Path]) -> List[Path]:
    """Gather candidate checkpoints from a stockpile of Run_* folders."""
    root = Path(os.path.expanduser(str(stockpile_dir))).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Stockpile directory not found: {root}")
    run_dirs = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("Run_")]
    if not run_dirs:
        run_dirs = [p for p in root.glob("**/Run_*") if p.is_dir()]
    ckpts: List[Path] = []
    for run_dir in run_dirs:
        fold_dir = run_dir / "Fold_1"
        candidates = [
            fold_dir / "chosen.pth",
            fold_dir / "chosen",
            fold_dir / "snn_best.pth",
            fold_dir / "best_snn.pth",
        ]
        ckpt = next((c for c in candidates if c.exists()), None)
        if ckpt is not None:
            ckpts.append(ckpt)
    return ckpts


def pick_random_stockpile_ckpt(stockpile_dir: Union[str, Path], seed: Optional[int] = None) -> Path:
    """Select a checkpoint from a stockpile directory (optionally seeded for reproducibility)."""
    ckpts = _collect_stockpile_checkpoints(stockpile_dir)
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found under {stockpile_dir}")
    # If seed is provided, pin the choice; otherwise use a fresh RNG so global seeds don't pin it.
    rng = random.Random(int(seed)) if seed is not None else random.Random()
    chosen = rng.choice(ckpts)
    print(f"[info] Using stockpile base SNN: {chosen}")
    return chosen


def load_base_snn_state(ckpt_path: Union[str, Path], train_flags: Optional[Dict[str, bool]] = None, settings: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(ckpt_path, map_location=device)
    params = ckpt['params'] if 'params' in ckpt else ckpt
    def P(t):
        t = t.to(device)
        return torch.nn.Parameter(t.clone().detach(), requires_grad=True)
    state = {
        "w1": P(params['w1']), "w2": P(params['w2']), "v1": P(params['v1']),
        "alpha_hetero_1": P(params.get('alpha_hetero_1', params.get('alpha'))),
        "beta_hetero_1":  P(params.get('beta_hetero_1',  params.get('beta'))),
        "thresholds_1":   P(params.get('thresholds_1',  params.get('threshold'))),
        "reset_1":        P(params.get('reset_1',       params.get('reset'))),
        "rest_1":         P(params.get('rest_1',        params.get('rest'))),
        "alpha_hetero_2": P(params.get('alpha_hetero_2', params.get('alpha_2'))),
        "beta_hetero_2":  P(params.get('beta_hetero_2',  params.get('beta_2'))),
    }
    if "input_delay_logits" in params:
        state["input_delay_logits"] = P(params["input_delay_logits"])
    elif settings is not None and settings.get("use_input_delay", False) and "nb_inputs" in settings:
        state["input_delay_logits"] = _init_input_delay_logits(settings, int(settings["nb_inputs"]))
    flags = _normalize_train_flags(train_flags)
    return _apply_snn_train_flags(state, flags)

def load_modulated_checkpoint(ckpt_path: Union[str, Path], settings: Dict, quiet: bool = False) -> Tuple[Dict[str, torch.Tensor], nn.Module]:
    ckpt = torch.load(ckpt_path, map_location=device)
    snn_params = ckpt.get("snn_params")
    mlp_state = ckpt.get("mlp_state")
    if snn_params is None or mlp_state is None:
        raise ValueError(f"Checkpoint at {ckpt_path} is missing SNN or MLP parameters.")
    state = {k: torch.nn.Parameter(v.to(device), requires_grad=False) for k, v in snn_params.items()}
    if settings.get("use_input_delay", False) and "input_delay_logits" not in state:
        state["input_delay_logits"] = _init_input_delay_logits(settings, int(settings["nb_inputs"]))
    mode = _select_mlp_mode(settings, ckpt.get("mlp_mode"))

    def _list_from(obj):
        if obj is None:
            return None
        if isinstance(obj, (list, tuple)):
            return [int(v) for v in obj]
        return None

    hidden_sizes = _resolve_mlp_hidden_sizes(settings)
    if hidden_sizes is None:
        hidden_sizes = _list_from(ckpt.get("mlp_hidden_sizes"))

    mod_hidden_sizes = settings.get("mod_hidden_sizes")
    if mod_hidden_sizes is None:
        mod_hidden_sizes = _list_from(ckpt.get("mod_hidden_sizes"))

    if settings.get("snn_mod_hidden") is None and ckpt.get("snn_mod_hidden") is not None:
        settings["snn_mod_hidden"] = ckpt.get("snn_mod_hidden")
    if settings.get("snn_mod_hidden_recurrent") is None:
        settings["snn_mod_hidden_recurrent"] = ckpt.get("snn_mod_hidden_recurrent", False)

    modulator = build_modulator(
        settings,
        override_mode=mode,
        hidden_sizes=mod_hidden_sizes if _is_snn_mode(mode) else hidden_sizes,
    )
    if not quiet:
        print(format_param_stats(state, modulator, prefix="load_modulated_checkpoint:", settings=settings))
    load_status = modulator.load_state_dict(mlp_state, strict=False)
    if not quiet:
        missing = load_status.missing_keys
        unexpected = load_status.unexpected_keys
        if missing:
            print(f"[warn] Missing keys when loading modulator: {missing}")
        if unexpected:
            print(f"[warn] Unexpected keys when loading modulator: {unexpected}")
    modulator.eval()
    return state, modulator

# -------------------------
# Modulated forward
# -------------------------
def run_snn_modulated(
    inputs,
    state: Dict,
    settings: Dict,
    modulator: nn.Module,
    mlp_interval: int,
    mlp_input_block_mask: Optional[Dict[str, bool]] = None,
    mlp_output_block_mask: Optional[Dict[str, bool]] = None,
    trace_fn: Optional[Callable[[int, Dict[str, torch.Tensor]], None]] = None,
    training: bool = False,
):
    """
    Core simulation loop. At each interval:
      - Gather recent spikes/params for the primary modulator (MLP or SNN).
      - If NM is enabled, the primary emits raw neuromodulator levels (linear head).
        The shared secondary NM mapper (sigmoid/tanh head) turns those into per-param
        values/deltas, which are then applied (substitution or additive).
      - Otherwise, apply the primary modulator outputs directly.
    """
    nb_hidden   = settings["nb_hidden"]
    nb_outputs  = settings["nb_outputs"]
    nb_steps    = settings["nb_steps"]
    psp_norm = bool(settings.get("psp_norm_peak", False))
    batch_size  = inputs.size(0)
    base_interval = max(1, int(mlp_interval))
    param_ranges = _resolve_param_ranges(settings)
    mlp_mode = _select_mlp_mode(settings)
    hidden_group_layout = _get_group_layout(settings, 0, nb_hidden)
    output_group_layout = _get_group_layout(settings, 1, nb_outputs)
    substitution_mode = _is_substitution_mode(mlp_mode)
    hidden_ann_layout = hidden_group_layout
    output_ann_layout = output_group_layout
    if substitution_mode:
        hidden_ann_layout = _uniform_group_layout(hidden_group_layout)
        output_ann_layout = _uniform_group_layout(output_group_layout)
    mod_mask = _build_fixed_mod_mask(settings, hidden_group_layout, output_group_layout)
    mod_mask_enabled = mod_mask is not None
    mod_hid_flat_group = bool(settings.get("mod_hid_flat_group", False))
    mod_hid_flat_mod_only = bool(settings.get("mod_hid_flat_modulated_only", False))
    param_names = MOD_TARGET_PARAM_NAMES
    timescale_ctrl = getattr(modulator, "param_timescales", None)
    intervals = {name: torch.tensor(float(base_interval), device=device, dtype=dtype) for name in param_names}
    if timescale_ctrl is not None:
        intervals.update({k: v.to(device=device, dtype=dtype) for k, v in timescale_ctrl.intervals().items()})
    update_ticks = {name: intervals[name].clone() for name in param_names}

    smoothing_ctrl = getattr(modulator, "param_smoothing", None)
    smoothing_cfg = settings.get("param_smoothing", {})
    smoothing_modes = set(smoothing_cfg.get("modes", []))
    smoothing_allowed = bool(smoothing_ctrl is not None) and bool(smoothing_cfg.get("enable")) and (
        not smoothing_modes or mlp_mode in smoothing_modes
    )
    smoothing_factors = {}
    if smoothing_allowed:
        smoothing_factors = {k: v.to(device=device, dtype=dtype) for k, v in smoothing_ctrl.mixing_factors().items()}
    smoothing_active = bool(smoothing_factors)

    mod_current_enable = bool(settings.get("mod_current_enable", False))
    raw_current_target = settings.get("mod_current_target", "both")
    def _normalize_current_target(val):
        key = str(val or "both").strip().lower()
        if key in {"hid", "hidden", "h"}:
            return "hidden"
        if key in {"out", "output", "o"}:
            return "output"
        if key in {"both", "all", "ho", "hidden_output", "hidden+output"}:
            return "both"
        return "both"
    mod_current_target = _normalize_current_target(raw_current_target)
    use_current_hidden = mod_current_enable and mod_current_target in {"hidden", "both"}
    use_current_output = mod_current_enable and mod_current_target in {"output", "both"}

    def _avg_mix(names: List[str]) -> Optional[torch.Tensor]:
        if not smoothing_active:
            return None
        mixes = [smoothing_factors[name] for name in names if name in smoothing_factors]
        if not mixes:
            return None
        return torch.stack(mixes, dim=0).mean(dim=0)

    hidden_current_mix = _avg_mix(HIDDEN_PARAM_NAMES) if (mod_current_enable and use_current_hidden) else None
    output_current_mix = _avg_mix(OUTPUT_PARAM_NAMES) if (mod_current_enable and use_current_output) else None

    w1 = state["w1"]; w2 = state["w2"]; v1 = state["v1"]
    a1 = state["alpha_hetero_1"]; b1 = state["beta_hetero_1"]
    thr = state["thresholds_1"];   rst = state["reset_1"]; rpo = state["rest_1"]
    a2 = state["alpha_hetero_2"];  b2 = state["beta_hetero_2"]

    alpha_1_local = a1.expand(batch_size, nb_hidden).clone()
    beta_1_local  = b1.expand(batch_size, nb_hidden).clone()
    thr_local     = thr.expand(batch_size, nb_hidden).clone()
    rst_local     = rst.expand(batch_size, nb_hidden).clone()
    rpo_local     = rpo.expand(batch_size, nb_hidden).clone()
    alpha_2_local = a2.expand(batch_size, nb_outputs).clone()
    beta_2_local  = b2.expand(batch_size, nb_outputs).clone()
    base_param_map = {
        "alpha_1": alpha_1_local.clone(),
        "beta_1": beta_1_local.clone(),
        "thr": thr_local.clone(),
        "reset": rst_local.clone(),
        "rest": rpo_local.clone(),
        "alpha_2": alpha_2_local.clone(),
        "beta_2": beta_2_local.clone(),
    }
    alpha_1_target = beta_1_target = thr_target = rst_target = rpo_target = alpha_2_target = beta_2_target = None
    if smoothing_active:
        alpha_1_target = alpha_1_local.clone()
        beta_1_target = beta_1_local.clone()
        thr_target = thr_local.clone()
        rst_target = rst_local.clone()
        rpo_target = rpo_local.clone()
        alpha_2_target = alpha_2_local.clone()
        beta_2_target = beta_2_local.clone()
    decay_dt = _effective_tau_decay_dt(settings)
    gain_hidden = None
    gain_out = None
    if psp_norm:
        gain_hidden = _psp_peak_gain(alpha_1_local, beta_1_local, dt=decay_dt)
        gain_out = _psp_peak_gain(alpha_2_local, beta_2_local, dt=decay_dt)

    hidden_current = None
    output_current = None
    hidden_current_target = None
    output_current_target = None
    if mod_current_enable:
        if use_current_hidden:
            hidden_current = torch.zeros((batch_size, nb_hidden), device=device, dtype=dtype)
            if smoothing_active:
                hidden_current_target = hidden_current.clone()
        if use_current_output:
            output_current = torch.zeros((batch_size, nb_outputs), device=device, dtype=dtype)
            if smoothing_active:
                output_current_target = output_current.clone()

    syn = torch.zeros((batch_size, nb_hidden), device=device, dtype=dtype)
    mem = torch.zeros((batch_size, nb_hidden), device=device, dtype=dtype)
    out_h = torch.zeros((batch_size, nb_hidden), device=device, dtype=dtype)
    hidden_dropout_p = float(settings.get("hidden_dropout_p", 0.0))

    delay_logits = state.get("input_delay_logits")
    max_delay = max(1, int(settings.get("input_delay_steps", 1))) if settings.get("use_input_delay", False) else 0
    inputs_delayed = _apply_input_delay(
        inputs, delay_logits, max_delay, temperature=settings.get("input_delay_temp", 1.0)
    )
    inputs_mod = inputs_delayed
    compress_mode = settings.get("channel_compress_mode")
    if compress_mode == "mod_only":
        factor = int(settings.get("channel_compress_factor_mod", settings.get("channel_compress_factor", 1)))
        if factor > 1:
            inputs_mod = compress_dense_inputs(inputs_delayed, factor, settings.get("nb_inputs_mod"))
    elif compress_mode == "mod_mlp":
        compressor = getattr(modulator, "input_compressor", None)
        if compressor is None:
            raise RuntimeError("channel_compress_mode=mod_mlp requires an input compressor on the modulator.")
        inputs_mod = compressor(inputs_delayed)
    h1_in = torch.einsum("btc,cd->btd", inputs_delayed, w1)  # [B,T,H]

    flt2 = torch.zeros((batch_size, nb_outputs), device=device, dtype=dtype)
    out2 = torch.zeros((batch_size, nb_outputs), device=device, dtype=dtype)
    out_rec = []

    mem_rec, spk_rec = [], []
    in_block_mask = mlp_input_block_mask or {}
    out_block_mask = mlp_output_block_mask or {}
    mlp_mode = _select_mlp_mode(settings)
    snn_mode = _is_snn_mode(mlp_mode)
    snn_add_mode = snn_mode and (mlp_mode == "snn_add")
    snn_sub_mode = snn_mode and (mlp_mode == "snn_sub")
    if snn_mode and not isinstance(modulator, (SNNAdditiveModulator, SNNSubstitutionModulator)):
        raise TypeError("snn mode requires an SNNAdditiveModulator or SNNSubstitutionModulator instance.")
    mlp = modulator if not snn_mode else None
    additive_mode = (mlp_mode == "mlp_add")
    combo_mode = (mlp_mode == "mlp_combo")
    combo_additive_list, combo_multiplicative_list = _resolve_ann_combo_lists(
        settings.get("ann_combo_additive"),
        settings.get("ann_combo_multiplicative"),
    )
    combo_additive = set(combo_additive_list)
    combo_multiplicative = set(combo_multiplicative_list)
    combo_activation_kind = (getattr(mlp, "output_activation_kind", "tanh") or "tanh").lower() if mlp is not None else "tanh"
    substitution_snn_values = False
    if mlp is not None and hasattr(mlp, "reset_sequence_state"):
        mlp.reset_sequence_state(batch_size, device=device, dtype=dtype)
    mlp_state_each_step = bool(settings.get("ann_rnn_state_every_step", False) and mlp is not None and getattr(mlp, "stateful", False))
    update_every_step = bool(settings.get("mod_update_every_step", False))

    if snn_mode:
        mod_hidden = modulator.mod_hidden
        init_param_groups: Optional[Dict[str, torch.Tensor]] = None
        if snn_sub_mode and isinstance(modulator, SNNSubstitutionModulator):
            def _group_param(name: str, tensor: torch.Tensor, layout: Optional['GroupLayout']):
                if layout is not None and (substitution_mode or mod_mask_enabled):
                    tensor = layout.project(tensor)
                if mod_mask_enabled and not getattr(modulator, "use_neuromodulators", False):
                    idx = mod_mask["hidden_union_idx"] if name in HIDDEN_PARAM_NAMES else mod_mask["output_union_idx"]
                    if idx is not None and idx.numel() > 0:
                        tensor = tensor.index_select(1, idx.to(device=device))
                return tensor
            init_param_groups = {
                "alpha_1": _group_param("alpha_1", alpha_1_local, hidden_group_layout),
                "beta_1": _group_param("beta_1", beta_1_local, hidden_group_layout),
                "thr": _group_param("thr", thr_local, hidden_group_layout),
                "reset": _group_param("reset", rst_local, hidden_group_layout),
                "rest": _group_param("rest", rpo_local, hidden_group_layout),
                "alpha_2": _group_param("alpha_2", alpha_2_local, output_group_layout),
                "beta_2": _group_param("beta_2", beta_2_local, output_group_layout),
            }
        mod_state = modulator.zero_state(batch_size, init_params=init_param_groups)
        mod_buffer = torch.zeros((base_interval, batch_size, mod_hidden), device=device, dtype=dtype) if snn_add_mode else None
        capture_mod_spikes = bool(training and settings.get("use_snn_mod_reg", False))
        mod_spk_rec = [] if capture_mod_spikes else None
        if capture_mod_spikes:
            setattr(modulator, "_last_spk_rec", None)

    def _apply_modulated_update(name: str, base_value: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
        if additive_mode:
            lo, hi = param_ranges.get(name, PARAMETER_RANGE_DEFAULTS.get(name, (-float("inf"), float("inf"))))
            lo_t = base_value.new_full(base_value.shape, float(lo))
            hi_t = base_value.new_full(base_value.shape, float(hi))
            headroom = torch.where(raw >= 0, hi_t - base_value, base_value - lo_t)
            delta = raw * headroom  # k=1 headroom scaling
            return _apply_additive_delta(base_value, delta, name, param_ranges)
        if substitution_mode:
            # Substitution expects values mapped through an activation into a known range.
            # In NM mode, let the NM mapper decide whether outputs are already squashed (e.g. sigmoid/tanh) or are logits.
            if getattr(mlp, "use_neuromodulators", False) and getattr(mlp, "nm_mapper", None) is not None:
                activation_kind = getattr(mlp.nm_mapper, "activation_kind", None) or "none"
            else:
                activation_kind = "sigmoid"
        else:
            activation_kind = getattr(mlp, "output_activation_kind", "sigmoid") if mlp is not None else "sigmoid"
        return _map_substitution_output(raw, name, activation_kind, param_ranges)

    def _aggregate_currents(effects: Dict[str, torch.Tensor], ready: Optional[Dict[str, torch.Tensor]]):
        if not effects:
            return None, None
        def _collect(names: List[str]) -> Optional[torch.Tensor]:
            parts = []
            for name in names:
                eff = effects.get(name)
                if eff is None:
                    continue
                if ready is not None and name in ready:
                    eff = eff * ready[name]
                parts.append(eff)
            if not parts:
                return None
            total = parts[0]
            for term in parts[1:]:
                total = total + term
            return total / float(len(parts))
        return _collect(HIDDEN_PARAM_NAMES), _collect(OUTPUT_PARAM_NAMES)

    def _update_mod_currents(effects: Dict[str, torch.Tensor], ready: Optional[Dict[str, torch.Tensor]]):
        nonlocal hidden_current, output_current, hidden_current_target, output_current_target
        if not mod_current_enable:
            return
        hid_cur, out_cur = _aggregate_currents(effects, ready)
        if use_current_hidden and hid_cur is not None:
            if smoothing_active and hidden_current_target is not None:
                hidden_current_target = hid_cur
            else:
                hidden_current = hid_cur
        if use_current_output and out_cur is not None:
            if smoothing_active and output_current_target is not None:
                output_current_target = out_cur
            else:
                output_current = out_cur

    def _expand_current_effect(vals: Optional[torch.Tensor], idx: Optional[torch.Tensor], full_dim: int) -> Optional[torch.Tensor]:
        if vals is None:
            return None
        if idx is None or vals.size(1) == full_dim:
            return vals
        out = vals.new_zeros((vals.size(0), full_dim))
        out.index_copy_(1, idx.to(device=vals.device), vals)
        return out

    def _normalize_combo_multiplier(raw: torch.Tensor) -> torch.Tensor:
        if combo_activation_kind == "sigmoid":
            raw = raw * 2.0 - 1.0
        return torch.clamp(raw, -1.0, 1.0)

    def _combo_target(name: str, base_value: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
        if name in combo_additive:
            lo, hi = param_ranges.get(name, PARAMETER_RANGE_DEFAULTS.get(name, (-float("inf"), float("inf"))))
            lo_t = base_value.new_full(base_value.shape, float(lo))
            hi_t = base_value.new_full(base_value.shape, float(hi))
            headroom = torch.where(raw >= 0, hi_t - base_value, base_value - lo_t)
            delta = raw * headroom
            return _apply_additive_delta(base_value, delta, name, param_ranges)
        lo, hi = param_ranges.get(name, PARAMETER_RANGE_DEFAULTS.get(name, (-float("inf"), float("inf"))))
        raw_norm = _normalize_combo_multiplier(raw)
        scale = torch.exp(raw_norm * COMBO_MULT_LOG_SPAN)
        proposed = base_value * scale
        return torch.clamp(proposed, min=float(lo), max=float(hi))

    def _apply_combo_update(
        name: str,
        current: torch.Tensor,
        raw: torch.Tensor,
        ready: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        base_value = base_param_map.get(name, current)
        proposed = _combo_target(name, base_value, raw)
        if target is not None:
            return proposed * ready + target * (1.0 - ready)
        return proposed * ready + current * (1.0 - ready)

    def _advance_param_timers() -> Dict[str, torch.Tensor]:
        ready = {}
        base_dt = torch.tensor(float(base_interval), device=device, dtype=dtype)
        for pname in param_names:
            update_ticks[pname] = update_ticks[pname] - base_dt
            mask = torch.sigmoid(-update_ticks[pname])
            # When mask ~1, add the interval back in; when ~0, just keep subtracting base_dt.
            update_ticks[pname] = update_ticks[pname] + intervals[pname] * mask
            ready[pname] = mask
        return ready

    def _smooth_param(current: torch.Tensor, target: torch.Tensor, name: str) -> torch.Tensor:
        mix = smoothing_factors.get(name)
        if mix is None:
            return target
        return current + mix * (target - current)

    def _prepare_grouped_input(tensor: torch.Tensor, layout: Optional[GroupLayout]) -> torch.Tensor:
        if layout is None:
            return tensor
        if substitution_mode or mod_mask_enabled:
            return layout.project(tensor)
        return tensor

    def _scale_headroom_effects_additive(effects: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if not effects:
            return effects
        scaled = {}
        cur_map = {
            "alpha_1": alpha_1_local,
            "beta_1": beta_1_local,
            "thr": thr_local,
            "reset": rst_local,
            "rest": rpo_local,
            "alpha_2": alpha_2_local,
            "beta_2": beta_2_local,
        }
        for name, eff in effects.items():
            cur = cur_map.get(name)
            if cur is None:
                scaled[name] = eff
                continue
            lo, hi = param_ranges.get(name, PARAMETER_RANGE_DEFAULTS.get(name, (-float("inf"), float("inf"))))
            lo_t = cur.new_full(cur.shape, float(lo))
            hi_t = cur.new_full(cur.shape, float(hi))
            headroom = torch.where(eff >= 0, hi_t - cur, cur - lo_t)
            scaled[name] = eff * headroom
        return scaled

    def _direct_substitute(name: str, current: torch.Tensor, proposed: torch.Tensor, ready: torch.Tensor) -> torch.Tensor:
        # Proposed values are already soft-bounded (e.g., tanh-mapped) by the modulator;
        # avoid an extra hard clamp here to preserve gradients and prevent "double clamp".
        return proposed * ready + current * (1.0 - ready)

    nm_cfg = settings.get("nm_cfg", {}) or {}
    def _get_pair(val, default):
        if isinstance(val, (list, tuple)):
            if len(val) >= 2:
                return (float(val[0]), float(val[1]))
            if len(val) == 1:
                return (float(val[0]), float(val[0]))
        try:
            v = float(val)
            return (v, v)
        except Exception:
            return default
    nm_neuron_frac = _get_pair(nm_cfg.get("neuron_fraction", (1.0, 1.0)), (1.0, 1.0))
    nm_param_frac_map = _parse_param_fraction_map(nm_cfg.get("param_fraction", None), default=1.0)
    nm_neuron_frac_enable = bool(nm_cfg.get("neuron_fraction_enable", False))
    nm_param_frac_enable = bool(nm_cfg.get("param_fraction_enable", False))
    hidden_target = hidden_ann_layout.group_count if hidden_ann_layout else nb_hidden
    output_target = output_ann_layout.group_count if output_ann_layout else nb_outputs
    mod_mask_tensors = {}
    if mod_mask_enabled:
        hidden_target = int(mod_mask["hidden_target"])
        output_target = int(mod_mask["output_target"])
        mod_mask_tensors["hidden_neuron_idx"] = mod_mask["hidden_neuron_idx"].to(device=device)
        mod_mask_tensors["output_neuron_idx"] = mod_mask["output_neuron_idx"].to(device=device)
        mod_mask_tensors["hidden_union_idx"] = mod_mask["hidden_union_idx"].to(device=device)
        mod_mask_tensors["output_union_idx"] = mod_mask["output_union_idx"].to(device=device)
        mod_mask_tensors["hidden_param_idx"] = {
            name: idx.to(device=device) for name, idx in mod_mask.get("hidden_param_idx", {}).items()
        }
        mod_mask_tensors["output_param_idx"] = {
            name: idx.to(device=device) for name, idx in mod_mask.get("output_param_idx", {}).items()
        }
        mod_mask_tensors["param_masks_target"] = {
            name: mask.to(device=device, dtype=dtype) for name, mask in mod_mask.get("param_masks_target", {}).items()
        }
        mod_mask_tensors["param_masks_union"] = {
            name: mask.to(device=device, dtype=dtype) for name, mask in mod_mask.get("param_masks_union", {}).items()
        }
        mod_mask_tensors["flat_hidden_idx"] = None
        mod_mask_tensors["flat_output_idx"] = None
        if mod_mask.get("flat_inputs"):
            if mod_mask.get("flat_hidden_idx") is not None:
                mod_mask_tensors["flat_hidden_idx"] = mod_mask["flat_hidden_idx"].to(device=device)
            if mod_mask.get("flat_output_idx") is not None:
                mod_mask_tensors["flat_output_idx"] = mod_mask["flat_output_idx"].to(device=device)

    def _apply_nm_masks(effects: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if mod_mask_enabled or not (nm_neuron_frac_enable or nm_param_frac_enable):
            return effects
        masked = {k: v for k, v in effects.items()}
        if nm_neuron_frac_enable:
            if "alpha_1" in masked:
                mask_h = (torch.rand((batch_size, masked["alpha_1"].size(1)), device=device, dtype=dtype) < float(nm_neuron_frac[0])).float()
                for key in ("alpha_1", "beta_1", "thr", "reset", "rest"):
                    if key in masked:
                        masked[key] = masked[key] * mask_h
            if "alpha_2" in masked:
                mask_o = (torch.rand((batch_size, masked["alpha_2"].size(1)), device=device, dtype=dtype) < float(nm_neuron_frac[1])).float()
                for key in ("alpha_2", "beta_2"):
                    if key in masked:
                        masked[key] = masked[key] * mask_o
        if nm_param_frac_enable:
            for key in ("alpha_1", "beta_1", "thr", "reset", "rest", "alpha_2", "beta_2"):
                if key in masked:
                    frac = float(nm_param_frac_map.get(key, 1.0))
                    masked[key] = masked[key] * (torch.rand_like(masked[key]) < frac).float()
        return masked

    def _expand_with_indices(values: torch.Tensor, indices: Optional[torch.Tensor], target_count: int) -> torch.Tensor:
        if indices is None:
            return values
        out = values.new_zeros((values.size(0), int(target_count)))
        if indices.numel() > 0:
            out[:, indices] = values
        return out

    def _expand_param_effect(name: str, values: torch.Tensor, layout: Optional[GroupLayout]) -> torch.Tensor:
        if not mod_mask_enabled:
            return layout.expand(values) if layout is not None else values
        target_count = hidden_target if name in HIDDEN_PARAM_NAMES else output_target
        idx_map = mod_mask_tensors.get("hidden_param_idx") if name in HIDDEN_PARAM_NAMES else mod_mask_tensors.get("output_param_idx")
        idx = idx_map.get(name) if idx_map else None
        expanded = _expand_with_indices(values, idx, target_count)
        mask = mod_mask_tensors.get("param_masks_target", {}).get(name)
        if mask is not None:
            expanded = expanded * mask
        if layout is not None:
            expanded = layout.expand(expanded)
        return expanded

    def _expand_nm_effect(name: str, values: torch.Tensor, layout: Optional[GroupLayout]) -> torch.Tensor:
        if not mod_mask_enabled:
            return layout.expand(values) if layout is not None else values
        target_count = hidden_target if name in HIDDEN_PARAM_NAMES else output_target
        idx = mod_mask_tensors.get("hidden_neuron_idx") if name in HIDDEN_PARAM_NAMES else mod_mask_tensors.get("output_neuron_idx")
        expanded = _expand_with_indices(values, idx, target_count)
        mask = mod_mask_tensors.get("param_masks_target", {}).get(name)
        if mask is not None:
            expanded = expanded * mask
        if layout is not None:
            expanded = layout.expand(expanded)
        return expanded

    def _expand_union_effect(name: str, values: torch.Tensor, layout: Optional[GroupLayout]) -> torch.Tensor:
        if not mod_mask_enabled:
            return layout.expand(values) if layout is not None else values
        target_count = hidden_target if name in HIDDEN_PARAM_NAMES else output_target
        idx = mod_mask_tensors.get("hidden_union_idx") if name in HIDDEN_PARAM_NAMES else mod_mask_tensors.get("output_union_idx")
        mask = mod_mask_tensors.get("param_masks_union", {}).get(name)
        if mask is not None:
            values = values * mask
        expanded = _expand_with_indices(values, idx, target_count)
        if layout is not None:
            expanded = layout.expand(expanded)
        return expanded

    def _select_param_input(name: str, tensor: torch.Tensor, layout: Optional[GroupLayout]) -> torch.Tensor:
        src = _prepare_grouped_input(tensor, layout)
        if not mod_mask_enabled:
            return src
        idx_map = mod_mask_tensors.get("hidden_param_idx") if name in HIDDEN_PARAM_NAMES else mod_mask_tensors.get("output_param_idx")
        idx = idx_map.get(name) if idx_map else None
        if idx is None:
            return src
        return src.index_select(1, idx)

    def _select_union_input(name: str, tensor: torch.Tensor, layout: Optional[GroupLayout]) -> torch.Tensor:
        src = _prepare_grouped_input(tensor, layout)
        if not mod_mask_enabled:
            return src
        idx = mod_mask_tensors.get("hidden_union_idx") if name in HIDDEN_PARAM_NAMES else mod_mask_tensors.get("output_union_idx")
        if idx is None:
            return src
        return src.index_select(1, idx)

    def _select_flat_input(tensor: torch.Tensor, idx: Optional[torch.Tensor]) -> torch.Tensor:
        if idx is None:
            return tensor
        return tensor.index_select(1, idx)

    def _expand_group_idx_to_neurons(
        layout: Optional['GroupLayout'],
        group_idx: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        if layout is None or getattr(layout, "forward", None) is None or group_idx is None:
            return group_idx
        if group_idx.numel() == 0:
            return group_idx
        weights = layout.forward.to(device=group_idx.device, dtype=dtype)
        try:
            selected = weights.index_select(0, group_idx)
        except Exception:
            return group_idx
        mask = selected.sum(dim=0) > 0
        return mask.nonzero(as_tuple=False).flatten().to(dtype=torch.long, device=group_idx.device)

    def _compute_mlp_output(t_step: int) -> torch.Tensor:
        if mlp is None:
            raise RuntimeError("MLP modulator not initialized.")
        if t_step >= base_interval:
            in_flat  = inputs_mod[:, t_step - base_interval + 1:t_step + 1, :].sum(dim=1)
            hid_flat = torch.stack(spk_rec[-base_interval:], dim=1).sum(dim=1)
            out_flat = torch.stack(out_rec[-base_interval:], dim=1).sum(dim=1)
        else:
            in_flat  = inputs_mod[:, :t_step + 1, :].sum(dim=1)
            hid_flat = torch.stack(spk_rec[:t_step + 1], dim=1).sum(dim=1)
            out_flat = torch.stack(out_rec[:t_step + 1], dim=1).sum(dim=1)

        parts = []
        slc = mlp.in_slices
        zero = lambda ref: torch.zeros_like(ref)
        if "alpha_1" in slc:
            src = _select_param_input("alpha_1", alpha_1_local, hidden_ann_layout)
            parts.append(src if in_block_mask.get("alpha_1", True) else zero(src))
        if "beta_1" in slc:
            src = _select_param_input("beta_1", beta_1_local, hidden_ann_layout)
            parts.append(src if in_block_mask.get("beta_1", True) else zero(src))
        if "thr" in slc:
            src = _select_param_input("thr", thr_local, hidden_ann_layout)
            parts.append(src if in_block_mask.get("thr", True) else zero(src))
        if "reset" in slc:
            src = _select_param_input("reset", rst_local, hidden_ann_layout)
            parts.append(src if in_block_mask.get("reset", True) else zero(src))
        if "rest" in slc:
            src = _select_param_input("rest", rpo_local, hidden_ann_layout)
            parts.append(src if in_block_mask.get("rest", True) else zero(src))
        if "alpha_2" in slc:
            src = _select_param_input("alpha_2", alpha_2_local, output_ann_layout)
            parts.append(src if in_block_mask.get("alpha_2", True) else zero(src))
        if "beta_2" in slc:
            src = _select_param_input("beta_2", beta_2_local, output_ann_layout)
            parts.append(src if in_block_mask.get("beta_2", True) else zero(src))
        if "in_flat" in slc:
            parts.append(in_flat if in_block_mask.get("in_flat", True) else zero(in_flat))
        if "hid_flat" in slc:
            src = hid_flat
            if mod_hid_flat_group and hidden_ann_layout is not None:
                src = hidden_ann_layout.project(src)
            if mod_hid_flat_mod_only and mod_mask_enabled:
                if mod_hid_flat_group:
                    idx = mod_mask_tensors.get("hidden_union_idx")
                else:
                    idx = _expand_group_idx_to_neurons(hidden_group_layout, mod_mask_tensors.get("hidden_union_idx"))
                src = _select_flat_input(src, idx)
            elif not (mod_hid_flat_group or mod_hid_flat_mod_only):
                src = _select_flat_input(src, mod_mask_tensors.get("flat_hidden_idx"))
            parts.append(src if in_block_mask.get("hid_flat", True) else zero(src))
        if "out_flat" in slc:
            src = _select_flat_input(out_flat, mod_mask_tensors.get("flat_output_idx"))
            parts.append(src if in_block_mask.get("out_flat", True) else zero(src))

        mlp_input = torch.cat(parts, dim=1)
        return mlp(mlp_input)

    nm_debug_print = bool(settings.get("nm_debug_print", False))
    nm_debug_done = False

    for t in range(nb_steps):
        h1_t = h1_in[:, t] + torch.einsum("bd,dc->bc", out_h, v1)  # [B,H]
        if mod_current_enable and use_current_hidden and hidden_current is not None:
            h1_t = h1_t + hidden_current

        mthr = mem - thr_local
        out_h = spike_fn(mthr)
        if hidden_dropout_p > 0.0 and training:
            out_h = F.dropout(out_h, p=hidden_dropout_p, training=True)
        rst_mask = (mthr > 0).float()

        syn = alpha_1_local * syn + h1_t
        syn_term = (1 - beta_1_local) * syn
        if gain_hidden is not None:
            syn_term = gain_hidden * syn_term
        mem = beta_1_local * (mem - rpo_local) + rpo_local + syn_term - rst_mask * (thr_local - rst_local)

        mem_rec.append(mem); spk_rec.append(out_h)

        h2_t = torch.einsum("bd,do->bo", out_h, w2)                # [B,O]
        if mod_current_enable and use_current_output and output_current is not None:
            h2_t = h2_t + output_current

        flt2 = alpha_2_local * flt2 + h2_t
        flt_term = (1 - beta_2_local) * flt2
        if gain_out is not None:
            flt_term = gain_out * flt_term
        out2 = beta_2_local * out2 + flt_term
        out_rec.append(out2)

        mlp_out_cached = None
        if mlp_state_each_step:
            mlp_out_cached = _compute_mlp_output(t)

        if snn_mode:
            alpha_1_in = alpha_1_local
            beta_1_in = beta_1_local
            thr_in = thr_local
            rst_in = rst_local
            rpo_in = rpo_local
            alpha_2_in = alpha_2_local
            beta_2_in = beta_2_local
            hid_spk_in = out_h
            out_spk_in = out2
            if mod_mask_enabled:
                alpha_1_in = _select_union_input("alpha_1", alpha_1_in, hidden_group_layout)
                beta_1_in = _select_union_input("beta_1", beta_1_in, hidden_group_layout)
                thr_in = _select_union_input("thr", thr_in, hidden_group_layout)
                rst_in = _select_union_input("reset", rst_in, hidden_group_layout)
                rpo_in = _select_union_input("rest", rpo_in, hidden_group_layout)
                alpha_2_in = _select_union_input("alpha_2", alpha_2_in, output_group_layout)
                beta_2_in = _select_union_input("beta_2", beta_2_in, output_group_layout)
                hid_spk_in = _select_flat_input(hid_spk_in, mod_mask_tensors.get("flat_hidden_idx"))
                out_spk_in = _select_flat_input(out_spk_in, mod_mask_tensors.get("flat_output_idx"))
            features = modulator.build_features(
                alpha_1_in, beta_1_in, thr_in, rst_in, rpo_in,
                alpha_2_in, beta_2_in,
                inputs_mod[:, t, :], hid_spk_in, out_spk_in, t
            )
            mod_spk, mod_state = modulator.forward_step(features, mod_state)
            if mod_buffer is not None:
                mod_buffer[t % base_interval] = mod_spk
            if mod_spk_rec is not None:
                mod_spk_rec.append(mod_spk)

        if snn_mode:
            do_update = update_every_step or ((t + 1) % base_interval == 0)
        else:
            do_update = update_every_step or (t % base_interval == 0)
        if do_update:
            if snn_mode:
                # Notebook behavior: update every interval with full-strength masks.
                ready_masks = {name: torch.ones_like(update_ticks[name]) for name in param_names}
            else:
                ready_masks = _advance_param_timers()

            if snn_mode:
                if update_every_step or (t + 1 >= base_interval):
                    if snn_add_mode and mod_buffer is not None:
                        span = max(1, int(min(t + 1, base_interval)))
                        recent_spikes = mod_buffer.sum(dim=0) / float(span)
                        effects = modulator.modulation_effects(recent_spikes)
                        expanded = {}
                        for key, vals in effects.items():
                            layout = hidden_group_layout if key in HIDDEN_PARAM_NAMES else output_group_layout if key in OUTPUT_PARAM_NAMES else None
                            if modulator.use_neuromodulators:
                                expanded[key] = _expand_nm_effect(key, vals, layout)
                            else:
                                expanded[key] = _expand_union_effect(key, vals, layout)
                        effects = expanded
                        if modulator.use_neuromodulators:
                            effects = _scale_headroom_effects_additive(effects)
                        effects = _apply_nm_masks(effects)
                        if mod_current_enable:
                            _update_mod_currents(effects, ready_masks)
                        else:
                            if "alpha_1" in ready_masks:
                                if smoothing_active:
                                    alpha_1_target = _apply_additive_delta(alpha_1_target, effects["alpha_1"] * ready_masks["alpha_1"], "alpha_1", param_ranges)
                                else:
                                    alpha_1_local = _apply_additive_delta(alpha_1_local, effects["alpha_1"] * ready_masks["alpha_1"], "alpha_1", param_ranges)
                                    if gain_hidden is not None:
                                        gain_hidden = _psp_peak_gain(alpha_1_local, beta_1_local, dt=decay_dt)
                            if "beta_1" in ready_masks:
                                if smoothing_active:
                                    beta_1_target = _apply_additive_delta(beta_1_target, effects["beta_1"] * ready_masks["beta_1"], "beta_1", param_ranges)
                                else:
                                    beta_1_local = _apply_additive_delta(beta_1_local, effects["beta_1"] * ready_masks["beta_1"], "beta_1", param_ranges)
                                    if gain_hidden is not None:
                                        gain_hidden = _psp_peak_gain(alpha_1_local, beta_1_local, dt=decay_dt)
                            if "alpha_2" in ready_masks:
                                if smoothing_active:
                                    alpha_2_target = _apply_additive_delta(alpha_2_target, effects["alpha_2"] * ready_masks["alpha_2"], "alpha_2", param_ranges)
                                else:
                                    alpha_2_local = _apply_additive_delta(alpha_2_local, effects["alpha_2"] * ready_masks["alpha_2"], "alpha_2", param_ranges)
                                    if gain_out is not None:
                                        gain_out = _psp_peak_gain(alpha_2_local, beta_2_local, dt=decay_dt)
                            if "beta_2" in ready_masks:
                                if smoothing_active:
                                    beta_2_target = _apply_additive_delta(beta_2_target, effects["beta_2"] * ready_masks["beta_2"], "beta_2", param_ranges)
                                else:
                                    beta_2_local = _apply_additive_delta(beta_2_local, effects["beta_2"] * ready_masks["beta_2"], "beta_2", param_ranges)
                                    if gain_out is not None:
                                        gain_out = _psp_peak_gain(alpha_2_local, beta_2_local, dt=decay_dt)
                            if "thr" in ready_masks:
                                if smoothing_active:
                                    thr_target = _apply_additive_delta(thr_target, effects["thr"] * ready_masks["thr"], "thr", param_ranges)
                                else:
                                    thr_local = _apply_additive_delta(thr_local, effects["thr"] * ready_masks["thr"], "thr", param_ranges)
                            if "reset" in ready_masks:
                                if smoothing_active:
                                    rst_target = _apply_additive_delta(rst_target, effects["reset"] * ready_masks["reset"], "reset", param_ranges)
                                else:
                                    rst_local = _apply_additive_delta(rst_local, effects["reset"] * ready_masks["reset"], "reset", param_ranges)
                            if "rest" in ready_masks:
                                if smoothing_active:
                                    rpo_target = _apply_additive_delta(rpo_target, effects["rest"] * ready_masks["rest"], "rest", param_ranges)
                                else:
                                    rpo_local = _apply_additive_delta(rpo_local, effects["rest"] * ready_masks["rest"], "rest", param_ranges)
                    elif snn_sub_mode:
                        out_mem = mod_state[-1].get("mem")
                        if out_mem is None:
                            continue
                        effects = modulator.modulation_effects(out_mem)
                        expanded = {}
                        for key, vals in effects.items():
                            layout = hidden_group_layout if key in HIDDEN_PARAM_NAMES else output_group_layout if key in OUTPUT_PARAM_NAMES else None
                            if modulator.use_neuromodulators:
                                expanded[key] = _expand_nm_effect(key, vals, layout)
                            else:
                                expanded[key] = _expand_union_effect(key, vals, layout)
                        effects = _apply_nm_masks(expanded)
                        if mod_current_enable:
                            _update_mod_currents(effects, ready_masks)
                        else:
                            direct_sub = not getattr(modulator, "use_neuromodulators", False)
                            if "alpha_1" in ready_masks:
                                if smoothing_active:
                                    if direct_sub:
                                        alpha_1_target = _direct_substitute("alpha_1", alpha_1_target, effects["alpha_1"], ready_masks["alpha_1"])
                                    else:
                                        alpha_1_target = _apply_modulated_update("alpha_1", alpha_1_target, effects["alpha_1"] * ready_masks["alpha_1"])
                                else:
                                    if direct_sub:
                                        alpha_1_local = _direct_substitute("alpha_1", alpha_1_local, effects["alpha_1"], ready_masks["alpha_1"])
                                    else:
                                        alpha_1_local = _apply_modulated_update("alpha_1", alpha_1_local, effects["alpha_1"] * ready_masks["alpha_1"])
                                    if gain_hidden is not None:
                                        gain_hidden = _psp_peak_gain(alpha_1_local, beta_1_local, dt=decay_dt)
                            if "beta_1" in ready_masks:
                                if smoothing_active:
                                    if direct_sub:
                                        beta_1_target = _direct_substitute("beta_1", beta_1_target, effects["beta_1"], ready_masks["beta_1"])
                                    else:
                                        beta_1_target = _apply_modulated_update("beta_1", beta_1_target, effects["beta_1"] * ready_masks["beta_1"])
                                else:
                                    if direct_sub:
                                        beta_1_local = _direct_substitute("beta_1", beta_1_local, effects["beta_1"], ready_masks["beta_1"])
                                    else:
                                        beta_1_local = _apply_modulated_update("beta_1", beta_1_local, effects["beta_1"] * ready_masks["beta_1"])
                                    if gain_hidden is not None:
                                        gain_hidden = _psp_peak_gain(alpha_1_local, beta_1_local, dt=decay_dt)
                            if "alpha_2" in ready_masks:
                                if smoothing_active:
                                    if direct_sub:
                                        alpha_2_target = _direct_substitute("alpha_2", alpha_2_target, effects["alpha_2"], ready_masks["alpha_2"])
                                    else:
                                        alpha_2_target = _apply_modulated_update("alpha_2", alpha_2_target, effects["alpha_2"] * ready_masks["alpha_2"])
                                else:
                                    if direct_sub:
                                        alpha_2_local = _direct_substitute("alpha_2", alpha_2_local, effects["alpha_2"], ready_masks["alpha_2"])
                                    else:
                                        alpha_2_local = _apply_modulated_update("alpha_2", alpha_2_local, effects["alpha_2"] * ready_masks["alpha_2"])
                                    if gain_out is not None:
                                        gain_out = _psp_peak_gain(alpha_2_local, beta_2_local, dt=decay_dt)
                            if "beta_2" in ready_masks:
                                if smoothing_active:
                                    if direct_sub:
                                        beta_2_target = _direct_substitute("beta_2", beta_2_target, effects["beta_2"], ready_masks["beta_2"])
                                    else:
                                        beta_2_target = _apply_modulated_update("beta_2", beta_2_target, effects["beta_2"] * ready_masks["beta_2"])
                                else:
                                    if direct_sub:
                                        beta_2_local = _direct_substitute("beta_2", beta_2_local, effects["beta_2"], ready_masks["beta_2"])
                                    else:
                                        beta_2_local = _apply_modulated_update("beta_2", beta_2_local, effects["beta_2"] * ready_masks["beta_2"])
                                    if gain_out is not None:
                                        gain_out = _psp_peak_gain(alpha_2_local, beta_2_local, dt=decay_dt)
                            if "thr" in ready_masks:
                                if smoothing_active:
                                    if direct_sub:
                                        thr_target = _direct_substitute("thr", thr_target, effects["thr"], ready_masks["thr"])
                                    else:
                                        thr_target = _apply_modulated_update("thr", thr_target, effects["thr"] * ready_masks["thr"])
                                else:
                                    if direct_sub:
                                        thr_local = _direct_substitute("thr", thr_local, effects["thr"], ready_masks["thr"])
                                    else:
                                        thr_local = _apply_modulated_update("thr", thr_local, effects["thr"] * ready_masks["thr"])
                            if "reset" in ready_masks:
                                if smoothing_active:
                                    if direct_sub:
                                        rst_target = _direct_substitute("reset", rst_target, effects["reset"], ready_masks["reset"])
                                    else:
                                        rst_target = _apply_modulated_update("reset", rst_target, effects["reset"] * ready_masks["reset"])
                                else:
                                    if direct_sub:
                                        rst_local = _direct_substitute("reset", rst_local, effects["reset"], ready_masks["reset"])
                                    else:
                                        rst_local = _apply_modulated_update("reset", rst_local, effects["reset"] * ready_masks["reset"])
                            if "rest" in ready_masks:
                                if smoothing_active:
                                    if direct_sub:
                                        rpo_target = _direct_substitute("rest", rpo_target, effects["rest"], ready_masks["rest"])
                                    else:
                                        rpo_target = _apply_modulated_update("rest", rpo_target, effects["rest"] * ready_masks["rest"])
                                else:
                                    if direct_sub:
                                        rpo_local = _direct_substitute("rest", rpo_local, effects["rest"], ready_masks["rest"])
                                    else:
                                        rpo_local = _apply_modulated_update("rest", rpo_local, effects["rest"] * ready_masks["rest"])
                            if (not smoothing_active) and psp_norm:
                                gain_hidden = _psp_peak_gain(alpha_1_local, beta_1_local, dt=decay_dt)
                                gain_out = _psp_peak_gain(alpha_2_local, beta_2_local, dt=decay_dt)
        if do_update and (not snn_mode):
            # ANN modes: fall through to MLP application below.
            mlp_out = mlp_out_cached if mlp_out_cached is not None else _compute_mlp_output(t)
            if mod_current_enable and getattr(mlp, "current_mode", False):
                slc_out = mlp.current_out_slices or mlp.out_slices
                hid_cur = out_cur = None
                if "hidden_current" in slc_out:
                    hid_cur = mlp_out[:, slc_out["hidden_current"]]
                if "output_current" in slc_out:
                    out_cur = mlp_out[:, slc_out["output_current"]]
                if mod_mask_enabled:
                    hid_cur = _expand_current_effect(hid_cur, mod_mask_tensors.get("hidden_union_idx"), nb_hidden)
                    out_cur = _expand_current_effect(out_cur, mod_mask_tensors.get("output_union_idx"), nb_outputs)
                if smoothing_active:
                    if use_current_hidden and hid_cur is not None:
                        hidden_current_target = hid_cur
                    if use_current_output and out_cur is not None:
                        output_current_target = out_cur
                else:
                    if use_current_hidden and hid_cur is not None:
                        hidden_current = hid_cur
                    if use_current_output and out_cur is not None:
                        output_current = out_cur
            elif getattr(mlp, "use_neuromodulators", False) and mlp.nm_mapper is not None:
                effects = mlp.nm_mapper.effects_from_flat(mlp_out)
                expanded = {}
                for key, vals in effects.items():
                    layout = hidden_ann_layout if key in HIDDEN_PARAM_NAMES else output_ann_layout if key in OUTPUT_PARAM_NAMES else None
                    expanded[key] = _expand_nm_effect(key, vals, layout)
                effects = expanded
                # Note: additive-mode headroom scaling is handled inside `_apply_modulated_update`.
                # If we scale here as well, deltas get headroom-scaled twice (and become too small).
                effects = _apply_nm_masks(effects)
                if nm_debug_print and (not nm_debug_done):
                    nm_debug_done = True
                    with torch.no_grad():
                        print(
                            f"[NM debug] t={t} mlp_out={tuple(mlp_out.shape)} "
                            f"flat_order={getattr(mlp.nm_mapper, 'flat_order', None)} mapper={getattr(mlp.nm_mapper, 'mapper_type', None)}"
                        )
                        for name in MOD_TARGET_PARAM_NAMES:
                            eff = effects.get(name)
                            if eff is None:
                                continue
                            v = eff.detach()
                            finite = torch.isfinite(v)
                            v_f = v[finite] if finite.any() else v.reshape(-1)[:0]
                            if v_f.numel() == 0:
                                print(f"[NM debug] {name}: all non-finite")
                                continue
                            print(
                                f"[NM debug] {name}: min={float(v_f.min()):.5g} max={float(v_f.max()):.5g} "
                                f"mean={float(v_f.mean()):.5g} std={float(v_f.std(unbiased=False)):.5g}"
                            )
                if mod_current_enable:
                    _update_mod_currents(effects, ready_masks)
                else:
                    if combo_mode:
                        if "alpha_1" in ready_masks:
                            if smoothing_active:
                                alpha_1_target = _apply_combo_update(
                                    "alpha_1", alpha_1_local, effects["alpha_1"], ready_masks["alpha_1"], target=alpha_1_target
                                )
                            else:
                                alpha_1_local = _apply_combo_update("alpha_1", alpha_1_local, effects["alpha_1"], ready_masks["alpha_1"])
                        if "beta_1" in ready_masks:
                            if smoothing_active:
                                beta_1_target = _apply_combo_update(
                                    "beta_1", beta_1_local, effects["beta_1"], ready_masks["beta_1"], target=beta_1_target
                                )
                            else:
                                beta_1_local = _apply_combo_update("beta_1", beta_1_local, effects["beta_1"], ready_masks["beta_1"])
                        if "thr" in ready_masks:
                            if smoothing_active:
                                thr_target = _apply_combo_update(
                                    "thr", thr_local, effects["thr"], ready_masks["thr"], target=thr_target
                                )
                            else:
                                thr_local = _apply_combo_update("thr", thr_local, effects["thr"], ready_masks["thr"])
                        if "reset" in ready_masks:
                            if smoothing_active:
                                rst_target = _apply_combo_update(
                                    "reset", rst_local, effects["reset"], ready_masks["reset"], target=rst_target
                                )
                            else:
                                rst_local = _apply_combo_update("reset", rst_local, effects["reset"], ready_masks["reset"])
                        if "rest" in ready_masks:
                            if smoothing_active:
                                rpo_target = _apply_combo_update(
                                    "rest", rpo_local, effects["rest"], ready_masks["rest"], target=rpo_target
                                )
                            else:
                                rpo_local = _apply_combo_update("rest", rpo_local, effects["rest"], ready_masks["rest"])
                        if "alpha_2" in ready_masks:
                            if smoothing_active:
                                alpha_2_target = _apply_combo_update(
                                    "alpha_2", alpha_2_local, effects["alpha_2"], ready_masks["alpha_2"], target=alpha_2_target
                                )
                            else:
                                alpha_2_local = _apply_combo_update("alpha_2", alpha_2_local, effects["alpha_2"], ready_masks["alpha_2"])
                        if "beta_2" in ready_masks:
                            if smoothing_active:
                                beta_2_target = _apply_combo_update(
                                    "beta_2", beta_2_local, effects["beta_2"], ready_masks["beta_2"], target=beta_2_target
                                )
                            else:
                                beta_2_local = _apply_combo_update("beta_2", beta_2_local, effects["beta_2"], ready_masks["beta_2"])
                        if (not smoothing_active) and psp_norm:
                            gain_hidden = _psp_peak_gain(alpha_1_local, beta_1_local, dt=decay_dt)
                            gain_out = _psp_peak_gain(alpha_2_local, beta_2_local, dt=decay_dt)
                    else:
                        if "alpha_1" in ready_masks:
                            if smoothing_active:
                                alpha_1_target = _apply_modulated_update("alpha_1", alpha_1_target, effects["alpha_1"] * ready_masks["alpha_1"])
                            else:
                                alpha_1_local = _apply_modulated_update("alpha_1", alpha_1_local, effects["alpha_1"] * ready_masks["alpha_1"])
                        if "beta_1" in ready_masks:
                            if smoothing_active:
                                beta_1_target = _apply_modulated_update("beta_1", beta_1_target, effects["beta_1"] * ready_masks["beta_1"])
                            else:
                                beta_1_local = _apply_modulated_update("beta_1", beta_1_local, effects["beta_1"] * ready_masks["beta_1"])
                        if "thr" in ready_masks:
                            if smoothing_active:
                                thr_target = _apply_modulated_update("thr", thr_target, effects["thr"] * ready_masks["thr"])
                            else:
                                thr_local = _apply_modulated_update("thr", thr_local, effects["thr"] * ready_masks["thr"])
                        if "reset" in ready_masks:
                            if smoothing_active:
                                rst_target = _apply_modulated_update("reset", rst_target, effects["reset"] * ready_masks["reset"])
                            else:
                                rst_local = _apply_modulated_update("reset", rst_local, effects["reset"] * ready_masks["reset"])
                        if "rest" in ready_masks:
                            if smoothing_active:
                                rpo_target = _apply_modulated_update("rest", rpo_target, effects["rest"] * ready_masks["rest"])
                            else:
                                rpo_local = _apply_modulated_update("rest", rpo_local, effects["rest"] * ready_masks["rest"])
                        if "alpha_2" in ready_masks:
                            if smoothing_active:
                                alpha_2_target = _apply_modulated_update("alpha_2", alpha_2_target, effects["alpha_2"] * ready_masks["alpha_2"])
                            else:
                                alpha_2_local = _apply_modulated_update("alpha_2", alpha_2_local, effects["alpha_2"] * ready_masks["alpha_2"])
                        if "beta_2" in ready_masks:
                            if smoothing_active:
                                beta_2_target = _apply_modulated_update("beta_2", beta_2_target, effects["beta_2"] * ready_masks["beta_2"])
                            else:
                                beta_2_local = _apply_modulated_update("beta_2", beta_2_local, effects["beta_2"] * ready_masks["beta_2"])
                        if (not smoothing_active) and psp_norm:
                            gain_hidden = _psp_peak_gain(alpha_1_local, beta_1_local, dt=decay_dt)
                            gain_out = _psp_peak_gain(alpha_2_local, beta_2_local, dt=decay_dt)
            else:
                slc_out = mlp.out_slices
                effects = {}
                if "alpha_1" in slc_out and out_block_mask.get("alpha_1", True):
                    s = slc_out["alpha_1"]; vals = mlp_out[:, s]
                    effects["alpha_1"] = _expand_param_effect("alpha_1", vals, hidden_ann_layout)
                if "beta_1" in slc_out and out_block_mask.get("beta_1", True):
                    s = slc_out["beta_1"]; vals = mlp_out[:, s]
                    effects["beta_1"] = _expand_param_effect("beta_1", vals, hidden_ann_layout)
                if "thr" in slc_out and out_block_mask.get("thr", True):
                    s = slc_out["thr"]; vals = mlp_out[:, s]
                    effects["thr"] = _expand_param_effect("thr", vals, hidden_ann_layout)
                if "reset" in slc_out and out_block_mask.get("reset", True):
                    s = slc_out["reset"]; vals = mlp_out[:, s]
                    effects["reset"] = _expand_param_effect("reset", vals, hidden_ann_layout)
                if "rest" in slc_out and out_block_mask.get("rest", True):
                    s = slc_out["rest"]; vals = mlp_out[:, s]
                    effects["rest"] = _expand_param_effect("rest", vals, hidden_ann_layout)
                if "alpha_2" in slc_out and out_block_mask.get("alpha_2", True):
                    s = slc_out["alpha_2"]; vals = mlp_out[:, s]
                    effects["alpha_2"] = _expand_param_effect("alpha_2", vals, output_ann_layout)
                if "beta_2" in slc_out and out_block_mask.get("beta_2", True):
                    s = slc_out["beta_2"]; vals = mlp_out[:, s]
                    effects["beta_2"] = _expand_param_effect("beta_2", vals, output_ann_layout)
                effects = _apply_nm_masks(effects)
                if mod_current_enable:
                    _update_mod_currents(effects, ready_masks)
                else:
                    if "alpha_1" in effects and "alpha_1" in ready_masks:
                        if combo_mode:
                            if smoothing_active:
                                alpha_1_target = _apply_combo_update(
                                    "alpha_1", alpha_1_local, effects["alpha_1"], ready_masks["alpha_1"], target=alpha_1_target
                                )
                            else:
                                alpha_1_local = _apply_combo_update("alpha_1", alpha_1_local, effects["alpha_1"], ready_masks["alpha_1"])
                        else:
                            if smoothing_active:
                                alpha_1_target = _apply_modulated_update("alpha_1", alpha_1_target, effects["alpha_1"] * ready_masks["alpha_1"])
                            else:
                                alpha_1_local = _apply_modulated_update("alpha_1", alpha_1_local, effects["alpha_1"] * ready_masks["alpha_1"])
                    if "beta_1" in effects and "beta_1" in ready_masks:
                        if combo_mode:
                            if smoothing_active:
                                beta_1_target = _apply_combo_update(
                                    "beta_1", beta_1_local, effects["beta_1"], ready_masks["beta_1"], target=beta_1_target
                                )
                            else:
                                beta_1_local = _apply_combo_update("beta_1", beta_1_local, effects["beta_1"], ready_masks["beta_1"])
                        else:
                            if smoothing_active:
                                beta_1_target = _apply_modulated_update("beta_1", beta_1_target, effects["beta_1"] * ready_masks["beta_1"])
                            else:
                                beta_1_local = _apply_modulated_update("beta_1", beta_1_local, effects["beta_1"] * ready_masks["beta_1"])
                    if "thr" in effects and "thr" in ready_masks:
                        if combo_mode:
                            if smoothing_active:
                                thr_target = _apply_combo_update("thr", thr_local, effects["thr"], ready_masks["thr"], target=thr_target)
                            else:
                                thr_local = _apply_combo_update("thr", thr_local, effects["thr"], ready_masks["thr"])
                        else:
                            if smoothing_active:
                                thr_target = _apply_modulated_update("thr", thr_target, effects["thr"] * ready_masks["thr"])
                            else:
                                thr_local = _apply_modulated_update("thr", thr_local, effects["thr"] * ready_masks["thr"])
                    if "reset" in effects and "reset" in ready_masks:
                        if combo_mode:
                            if smoothing_active:
                                rst_target = _apply_combo_update(
                                    "reset", rst_local, effects["reset"], ready_masks["reset"], target=rst_target
                                )
                            else:
                                rst_local = _apply_combo_update("reset", rst_local, effects["reset"], ready_masks["reset"])
                        else:
                            if smoothing_active:
                                rst_target = _apply_modulated_update("reset", rst_target, effects["reset"] * ready_masks["reset"])
                            else:
                                rst_local = _apply_modulated_update("reset", rst_local, effects["reset"] * ready_masks["reset"])
                    if "rest" in effects and "rest" in ready_masks:
                        if combo_mode:
                            if smoothing_active:
                                rpo_target = _apply_combo_update("rest", rpo_local, effects["rest"], ready_masks["rest"], target=rpo_target)
                            else:
                                rpo_local = _apply_combo_update("rest", rpo_local, effects["rest"], ready_masks["rest"])
                        else:
                            if smoothing_active:
                                rpo_target = _apply_modulated_update("rest", rpo_target, effects["rest"] * ready_masks["rest"])
                            else:
                                rpo_local = _apply_modulated_update("rest", rpo_local, effects["rest"] * ready_masks["rest"])
                    if "alpha_2" in effects and "alpha_2" in ready_masks:
                        if combo_mode:
                            if smoothing_active:
                                alpha_2_target = _apply_combo_update(
                                    "alpha_2", alpha_2_local, effects["alpha_2"], ready_masks["alpha_2"], target=alpha_2_target
                                )
                            else:
                                alpha_2_local = _apply_combo_update("alpha_2", alpha_2_local, effects["alpha_2"], ready_masks["alpha_2"])
                        else:
                            if smoothing_active:
                                alpha_2_target = _apply_modulated_update("alpha_2", alpha_2_target, effects["alpha_2"] * ready_masks["alpha_2"])
                            else:
                                alpha_2_local = _apply_modulated_update("alpha_2", alpha_2_local, effects["alpha_2"] * ready_masks["alpha_2"])
                    if "beta_2" in effects and "beta_2" in ready_masks:
                        if combo_mode:
                            if smoothing_active:
                                beta_2_target = _apply_combo_update(
                                    "beta_2", beta_2_local, effects["beta_2"], ready_masks["beta_2"], target=beta_2_target
                                )
                            else:
                                beta_2_local = _apply_combo_update("beta_2", beta_2_local, effects["beta_2"], ready_masks["beta_2"])
                        else:
                            if smoothing_active:
                                beta_2_target = _apply_modulated_update("beta_2", beta_2_target, effects["beta_2"] * ready_masks["beta_2"])
                            else:
                                beta_2_local = _apply_modulated_update("beta_2", beta_2_local, effects["beta_2"] * ready_masks["beta_2"])

        if smoothing_active:
            alpha_1_local = _smooth_param(alpha_1_local, alpha_1_target, "alpha_1")
            beta_1_local = _smooth_param(beta_1_local, beta_1_target, "beta_1")
            thr_local = _smooth_param(thr_local, thr_target, "thr")
            rst_local = _smooth_param(rst_local, rst_target, "reset")
            rpo_local = _smooth_param(rpo_local, rpo_target, "rest")
            alpha_2_local = _smooth_param(alpha_2_local, alpha_2_target, "alpha_2")
            beta_2_local = _smooth_param(beta_2_local, beta_2_target, "beta_2")
            if psp_norm:
                gain_hidden = _psp_peak_gain(alpha_1_local, beta_1_local, dt=decay_dt)
                gain_out = _psp_peak_gain(alpha_2_local, beta_2_local, dt=decay_dt)
            if mod_current_enable:
                if use_current_hidden and hidden_current_target is not None:
                    if hidden_current_mix is None:
                        hidden_current = hidden_current_target
                    else:
                        hidden_current = hidden_current + hidden_current_mix * (hidden_current_target - hidden_current)
                if use_current_output and output_current_target is not None:
                    if output_current_mix is None:
                        output_current = output_current_target
                    else:
                        output_current = output_current + output_current_mix * (output_current_target - output_current)

        if trace_fn is not None:
            payload = {
                "alpha_1": alpha_1_local.detach(),
                "beta_1": beta_1_local.detach(),
                "thr": thr_local.detach(),
                "reset": rst_local.detach(),
                "rest": rpo_local.detach(),
                "alpha_2": alpha_2_local.detach(),
                "beta_2": beta_2_local.detach(),
            }
            if smoothing_active:
                payload.update({
                    "alpha_1_target": alpha_1_target.detach(),
                    "beta_1_target": beta_1_target.detach(),
                    "thr_target": thr_target.detach(),
                    "reset_target": rst_target.detach(),
                    "rest_target": rpo_target.detach(),
                    "alpha_2_target": alpha_2_target.detach(),
                    "beta_2_target": beta_2_target.detach(),
                })
            trace_fn(t, payload)

    mem_rec = torch.stack(mem_rec, dim=1)      # [B,T,H]
    spk_rec = torch.stack(spk_rec, dim=1)      # [B,T,H]
    out_rec = torch.stack(out_rec, dim=1)      # [B,T,O]
    if snn_mode and training and mod_spk_rec is not None:
        setattr(modulator, "_last_spk_rec", torch.stack(mod_spk_rec, dim=1))  # [B,T,M]
    return out_rec, (mem_rec, spk_rec)

# -------------------------
# Eval helpers (Modulated)
# -------------------------
@torch.no_grad()
def eval_val_metrics_mod(state, settings, x_data, y_data,
                         modulator: nn.Module, mlp_interval: int,
                         indices: Optional[np.ndarray] = None,
                         mlp_input_block_mask: Optional[Dict[str, bool]] = None,
                         mlp_output_block_mask: Optional[Dict[str, bool]] = None):
    loss_fn = nn.NLLLoss(reduction="mean"); log_softmax = nn.LogSoftmax(dim=1)
    nlls, accs = [], []
    gen = sparse_data_generator_from_hdf5_spikes(
        x_data, y_data, settings["batch_size"], settings["nb_steps"],
        settings["nb_inputs"], settings["max_time"], shuffle=False, indices=indices,
        augment_cfg=settings.get("augment_eval"), postbin_time_mask=settings.get("postbin_mask_eval", 0.0)
    )
    for x_local, y_local in gen:
        out, _ = run_snn_modulated(
            x_local.to_dense(), state, settings, modulator, mlp_interval,
            mlp_input_block_mask=mlp_input_block_mask,
            mlp_output_block_mask=mlp_output_block_mask
        )
        m, _ = torch.max(out, 1)
        log_p = log_softmax(m)
        nlls.append(loss_fn(log_p, y_local).item())
        _, am = torch.max(m, 1)
        accs.append(torch.mean((y_local == am).float()).item())
    return float(np.mean(nlls)) if nlls else float("nan"), float(np.mean(accs)) if accs else 0.0

@torch.no_grad()
def evaluate_testset_mod(state, settings, x_test, y_test, modulator: nn.Module, mlp_interval: int,
                         mlp_input_block_mask: Optional[Dict[str, bool]] = None,
                         mlp_output_block_mask: Optional[Dict[str, bool]] = None):
    loss_fn = nn.NLLLoss(reduction="mean"); log_softmax = nn.LogSoftmax(dim=1)
    nlls, accs = [], []
    gen = sparse_data_generator_from_hdf5_spikes(
        x_test, y_test, settings["batch_size"], settings["nb_steps"],
        settings["nb_inputs"], settings["max_time"], shuffle=False,
        augment_cfg=settings.get("augment_test"),
        postbin_time_mask=settings.get("postbin_mask_test", 0.0)
    )
    for x_local, y_local in gen:
        out, _ = run_snn_modulated(
            x_local.to_dense(), state, settings, modulator, mlp_interval,
            mlp_input_block_mask=mlp_input_block_mask,
            mlp_output_block_mask=mlp_output_block_mask
        )
        m, _ = torch.max(out, 1)
        log_p = log_softmax(m)
        nlls.append(loss_fn(log_p, y_local).item())
        _, am = torch.max(m, 1)
        accs.append(torch.mean((y_local == am).float()).item())
    test_nll = float(np.mean(nlls)) if nlls else float("nan")
    test_acc = float(np.mean(accs)) if accs else 0.0
    print(f"Final Test (Mod) — NLL: {test_nll:.5f}, Accuracy: {test_acc:.5f}")
    return test_nll, test_acc

def _summarize_spike_activity(total_spikes: float, num_samples: int, nb_hidden: int) -> Dict[str, float]:
    num_samples = max(1, int(num_samples))
    nb_hidden = max(1, int(nb_hidden or 1))
    avg_per_sample = float(total_spikes) / num_samples
    avg_per_neuron = avg_per_sample / nb_hidden
    return {
        "total_spikes": float(total_spikes),
        "num_samples": num_samples,
        "avg_per_sample": avg_per_sample,
        "avg_per_neuron": avg_per_neuron,
    }

def _format_spike_stats(stats: Dict[str, float]) -> str:
    return (
        f"avg/sample={stats['avg_per_sample']:.4f}, avg/neuron={stats['avg_per_neuron']:.6f}, "
        f"samples={stats['num_samples']}, total_spikes={stats['total_spikes']:.2f}"
    )

@torch.no_grad()
def compute_snn_test_spike_stats(state, settings, x_test, y_test) -> Dict[str, float]:
    total_spikes = 0.0
    num_samples = 0
    gen = sparse_data_generator_from_hdf5_spikes(
        x_test, y_test, settings["batch_size"], settings["nb_steps"],
        settings["nb_inputs"], settings["max_time"], shuffle=False,
        augment_cfg=settings.get("augment_test"),
        postbin_time_mask=settings.get("postbin_mask_test", 0.0)
    )
    for x_local, _ in gen:
        batch_inputs = x_local.to_dense()
        num_samples += batch_inputs.size(0)
        _, (_, spk_rec) = run_snn_hetero(batch_inputs, state, settings)
        total_spikes += float(spk_rec.sum().item())
    return _summarize_spike_activity(total_spikes, num_samples, settings.get("nb_hidden", 1))

@torch.no_grad()
def compute_mod_test_spike_stats(
    state,
    settings,
    modulator: nn.Module,
    mlp_interval: int,
    x_test,
    y_test,
    mlp_input_block_mask: Optional[Dict[str, bool]] = None,
    mlp_output_block_mask: Optional[Dict[str, bool]] = None,
) -> Dict[str, float]:
    total_spikes = 0.0
    num_samples = 0
    modulator.eval()
    gen = sparse_data_generator_from_hdf5_spikes(
        x_test, y_test, settings["batch_size"], settings["nb_steps"],
        settings["nb_inputs"], settings["max_time"], shuffle=False,
        augment_cfg=settings.get("augment_test"),
        postbin_time_mask=settings.get("postbin_mask_test", 0.0)
    )
    for x_local, _ in gen:
        batch_inputs = x_local.to_dense()
        num_samples += batch_inputs.size(0)
        _, (_, spk_rec) = run_snn_modulated(
            batch_inputs, state, settings, modulator, mlp_interval,
            mlp_input_block_mask=mlp_input_block_mask,
            mlp_output_block_mask=mlp_output_block_mask,
        )
        total_spikes += float(spk_rec.sum().item())
    return _summarize_spike_activity(total_spikes, num_samples, settings.get("nb_hidden", 1))

def _collect_eval_batches(x_data, y_data, settings, batch_limit: int,
                          subset: Optional[np.ndarray] = None,
                          augment_cfg: Optional[dict] = None,
                          postbin_time_mask: float = 0.0):
    """
    Materialize a capped list of dense batches (CPU) for repeated evaluation sweeps.
    """
    batches = []
    gen = sparse_data_generator_from_hdf5_spikes(
        x_data, y_data, settings["batch_size"], settings["nb_steps"],
        settings["nb_inputs"], settings["max_time"], shuffle=False, indices=subset,
        augment_cfg=augment_cfg, postbin_time_mask=postbin_time_mask
    )
    for i, (x_sp, y) in enumerate(gen):
        batches.append((x_sp.to_dense().cpu(), y.cpu()))
        if batch_limit and (i + 1) >= batch_limit:
            break
    return batches

@torch.no_grad()
def _compute_mod_metric_from_batches(
    state,
    settings,
    modulator: nn.Module,
    mlp_interval: int,
    batches: List[Tuple[torch.Tensor, torch.Tensor]],
    metric: str = "acc",
    mlp_input_block_mask: Optional[Dict[str, bool]] = None,
    mlp_output_block_mask: Optional[Dict[str, bool]] = None,
):
    loss_fn = nn.NLLLoss(reduction="mean")
    log_softmax = nn.LogSoftmax(dim=1)
    total, correct = 0, 0
    losses = []
    for x_cpu, y_cpu in batches:
        x = x_cpu.to(device)
        y = y_cpu.to(device)
        out, _ = run_snn_modulated(
            x, state, settings, modulator, mlp_interval,
            mlp_input_block_mask=mlp_input_block_mask,
            mlp_output_block_mask=mlp_output_block_mask
        )
        m, _ = torch.max(out, 1)
        log_p = log_softmax(m)
        loss = loss_fn(log_p, y)
        losses.append(loss.item())
        pred = torch.argmax(m, dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    if metric == "nll":
        return float(np.mean(losses)) if losses else float("nan")
    return correct / max(1, total)

def _approximate_block_shapley(block_names: List[str], metric_fn, nb_samples: int):
    """
    Monte Carlo permutation-based Shapley approximation for a small feature set.
    """
    if not block_names:
        return {}, None
    baseline_mask = {b: False for b in block_names}
    baseline_metric = metric_fn(dict(baseline_mask))
    contribs = {b: 0.0 for b in block_names}
    for _ in range(max(1, nb_samples)):
        perm = block_names[:]
        random.shuffle(perm)
        mask = dict(baseline_mask)
        prev_metric = baseline_metric
        for blk in perm:
            mask[blk] = True
            metric_val = metric_fn(dict(mask))
            contribs[blk] += metric_val - prev_metric
            prev_metric = metric_val
    scale = 1.0 / max(1, nb_samples)
    for blk in contribs:
        contribs[blk] *= scale
    return contribs, baseline_metric

def run_mlp_block_shap_analysis(
    state,
    settings,
    mlp: ModulatingMLP,
    x_train, y_train, x_test, y_test,
    dataset: str = "test",
    nb_samples: int = 16,
    batch_limit: int = 4,
    metric: str = "acc",
    run_dir: Optional[Union[str, Path]] = None,
):
    """
    Approximate Shapley contributions for enabled MLP input/output blocks by
    toggling them and measuring the chosen metric over a fixed mini-dataset.
    """
    dataset = (dataset or "test").lower()
    if dataset not in {"train", "test"}:
        raise ValueError(f"Unsupported SHAP dataset '{dataset}'. Use 'train' or 'test'.")

    if dataset == "train":
        src_x, src_y = x_train, y_train
        augment_cfg = settings.get("augment_eval") if settings.get("augment_eval") else None
        postbin_mask = settings.get("postbin_mask_eval", 0.0)
    else:
        src_x, src_y = x_test, y_test
        augment_cfg = settings.get("augment_test") if settings.get("augment_test") else None
        postbin_mask = settings.get("postbin_mask_test", 0.0)

    batches = _collect_eval_batches(
        src_x, src_y, settings, batch_limit=batch_limit,
        subset=None, augment_cfg=augment_cfg, postbin_time_mask=postbin_mask
    )
    if not batches:
        print("SHAP analysis skipped: no evaluation batches could be collected.")
        return None

    metric = metric.lower()
    metric_sign = -1.0 if metric == "nll" else 1.0  # convert to maximization score
    def _score_to_metric(val: float) -> float:
        return val / metric_sign if metric_sign != 0 else val

    full_metric = _compute_mod_metric_from_batches(
        state, settings, mlp, settings["mlp_interval"], batches,
        metric=metric
    )

    reports = {"dataset": dataset, "metric": metric,
               "full_metric": full_metric, "input": None, "output": None}

    in_blocks = list(mlp.in_slices.keys())
    if in_blocks:
        def score_fn(mask_dict):
            return metric_sign * _compute_mod_metric_from_batches(
                state, settings, mlp, settings["mlp_interval"], batches,
                metric=metric,
                mlp_input_block_mask=mask_dict, mlp_output_block_mask=None
            )
        contribs_score, baseline_score = _approximate_block_shapley(in_blocks, score_fn, nb_samples)
        contribs_metric = {k: _score_to_metric(v) for k, v in contribs_score.items()}
        baseline_metric = _score_to_metric(baseline_score)
        reports["input"] = {
            "blocks": sorted(
                [{"name": k, "delta": contribs_metric[k]} for k in contribs_metric],
                key=lambda itm: abs(itm["delta"]), reverse=True
            ),
            "all_disabled_metric": baseline_metric
        }

    out_blocks = list(mlp.out_slices.keys())
    if out_blocks:
        def score_fn_out(mask_dict):
            return metric_sign * _compute_mod_metric_from_batches(
                state, settings, mlp, settings["mlp_interval"], batches,
                metric=metric,
                mlp_input_block_mask=None, mlp_output_block_mask=mask_dict
            )
        contribs_score, baseline_score = _approximate_block_shapley(out_blocks, score_fn_out, nb_samples)
        contribs_metric = {k: _score_to_metric(v) for k, v in contribs_score.items()}
        baseline_metric = _score_to_metric(baseline_score)
        reports["output"] = {
            "blocks": sorted(
                [{"name": k, "delta": contribs_metric[k]} for k in contribs_metric],
                key=lambda itm: abs(itm["delta"]), reverse=True
            ),
            "all_disabled_metric": baseline_metric
        }

    if run_dir:
        out_path = Path(run_dir) / "mlp_block_shap.json"
        with open(out_path, "w") as fh:
            json.dump(reports, fh, indent=2)
        print(f"Saved MLP block SHAP report to {out_path}")

    return reports

def _pretty_print_shap_report(report: Optional[dict]):
    if not report:
        return
    metric = report["metric"]
    print("\n===== MLP Block SHAP =====")
    print(f"Dataset: {report['dataset']} | Metric: {metric} | All blocks metric: {report['full_metric']:.5f}")
    def _log_section(name, payload):
        if not payload:
            return
        print(f"Top {name} blocks (Δ{metric} relative to disabled baseline {payload['all_disabled_metric']:.5f}):")
        for entry in payload["blocks"]:
            print(f"  - {entry['name']}: {entry['delta']:+.5f}")
    _log_section("input", report.get("input"))
    _log_section("output", report.get("output"))

def maybe_run_ann_shap(
    training_result: dict,
    settings_mod: dict,
    shap_cfg: dict,
    x_train, y_train, x_test, y_test
):
    if not shap_cfg.get("enable"):
        return
    if _is_snn_mode(_select_mlp_mode(settings_mod)):
        print("SHAP skipped: snn_add/snn_sub modes do not expose MLP blocks.")
        return
    if training_result is None or "run_dir" not in training_result:
        print("SHAP skipped: training result missing run_dir.")
        return
    fold_summary = None
    fold_summaries = training_result.get("fold_summaries")
    if isinstance(fold_summaries, list) and fold_summaries:
        fold_summary = fold_summaries[0]
    try:
        ckpt_path = _select_mod_ckpt(training_result["run_dir"], fold_summary)
    except Exception as exc:
        print(f"SHAP skipped: {exc}")
        return

    try:
        state_eval, mlp_eval = load_modulated_checkpoint(ckpt_path, settings_mod)
    except Exception as exc:
        print(f"SHAP skipped (failed to load checkpoint): {exc}")
        return

    report = run_mlp_block_shap_analysis(
        state_eval, settings_mod, mlp_eval,
        x_train, y_train, x_test, y_test,
        dataset=shap_cfg.get("dataset", "test"),
        nb_samples=shap_cfg.get("samples", 16),
        batch_limit=shap_cfg.get("batch_limit", 4),
        metric=shap_cfg.get("metric", "acc"),
        run_dir=training_result["run_dir"]
    )
    _pretty_print_shap_report(report)

def _select_mod_ckpt(run_dir: Union[str, Path], fold_summary: Optional[dict] = None):
    """
    Locate a reasonable checkpoint artifact inside a mod run directory.
    """
    run_dir = Path(run_dir)
    fold_id = fold_summary["fold"] if fold_summary and "fold" in fold_summary else 1
    fold_name = f"Fold_{fold_id}" if (run_dir / f"Fold_{fold_id}").exists() else "Fold_1"
    fold_dir = run_dir / fold_name
    candidates = [
        fold_dir / "mod_best.pth",
        fold_dir / "best_mod.pth",
        fold_dir / "chosen.pth",
        fold_dir / "chosen",
    ]
    best_epoch = None
    if fold_summary:
        best_epoch = fold_summary.get("best_epoch")
        if best_epoch is not None and best_epoch > 0:
            candidates.append(fold_dir / f"mod_{best_epoch}.pth")
    mod_ckpts = sorted(fold_dir.glob("mod_*.pth"), reverse=True)
    candidates.extend(mod_ckpts)
    for cand in candidates:
        if cand.exists():
            return cand
    raise FileNotFoundError(f"No modulation checkpoint found under {fold_dir}")


# -------------------------
# Train Modulated SNN
# -------------------------
def train_modulated_snn(
    state: Dict, settings: Dict, x_train, y_train, x_test, y_test,
    modulator: Optional[nn.Module], mlp_interval: int,
    x_val=None, y_val=None,
    save_every_epoch: bool = True,
    use_validation: bool = True, val_fraction: float = 0.1,
    k_folds: int = 1, patience: int = 20, seed: int = 0,
    fixed_split_path: Optional[str] = None,
    reinit_per_fold: bool = True,
    start_locked: bool = False, unlock_metric: Optional[str] = None, unlock_threshold: Optional[float] = None,
    base_ckpt_path: Optional[Union[str, Path]] = None,
    run_index: Optional[int] = None,
):
    if modulator is not None:
        settings["mlp_init_state_dict"] = copy.deepcopy(modulator.state_dict())

    run_dir = ensure_run_dir(settings["save_dir"], run_index=run_index)
    print(f"Saving to: {run_dir}")
    log_path = run_dir / "training_log.txt"
    log_file = open(log_path, "a", buffering=1)
    print(f"Logging to: {log_path}")
    def log(msg): print(msg) or print(msg, file=log_file)

    snn_disabled = sorted([k for k, v in (settings.get("snn_train_flags") or {}).items() if not v])

    external_val_available = (x_val is not None) and (y_val is not None)
    val_dataset_name = settings.get("val_file")
    use_validation_flag = use_validation or external_val_available
    if external_val_available:
        try:
            val_size = len(y_val)
        except TypeError:
            val_size = "unknown"
        log(f"[info] External validation split detected ({val_dataset_name or 'custom'}), size={val_size}.")

    log_run_header(
        log, run_dir, settings,
        mode="mod",
        use_validation=use_validation_flag,
        val_fraction=val_fraction,
        k_folds=k_folds,
        patience=patience,
        seed=seed,
        fixed_split_path=fixed_split_path,
        reinit_per_fold=reinit_per_fold,
        start_locked=start_locked,
        unlock_metric=unlock_metric,
        unlock_threshold=unlock_threshold,
        mlp_interval=mlp_interval,
        use_input_delay=settings.get("use_input_delay"),
        input_delay_steps=settings.get("input_delay_steps"),
        mod_reg_enabled=settings.get("use_mod_reg"),
        snn_mod_reg_enabled=settings.get("use_snn_mod_reg"),
        snn_train_disabled=snn_disabled,
        external_val=external_val_available,
        external_val_file=val_dataset_name,
    )

    # NEW: print base SNN ckpt metrics if provided (unchanged)
    if base_ckpt_path is not None:
        try:
            ckpt = torch.load(base_ckpt_path, map_location="cpu")
            epoch = ckpt.get("epoch", None)
            metrics = ckpt.get("metrics", None)
            history = ckpt.get("history", None)
            log("---- Base SNN checkpoint info ----")
            log(f"Path: {base_ckpt_path}")
            if epoch is not None:
                log(f"Epoch: {epoch}")
            if metrics:
                for k in ["train_loss","train_acc","val_nll","val_acc","test_nll","test_acc"]:
                    if k in metrics:
                        v = metrics[k]
                        log(f"{k}: {v:.5f}" if isinstance(v, (float,int)) else f"{k}: {v}")
            elif history:
                def _last(lst): 
                    return lst[-1] if isinstance(lst, list) and len(lst)>0 else None
                tl = _last(history.get("train_loss", []))
                ta = _last(history.get("train_acc", []))
                vl = _last(history.get("val_nll", []))
                va = _last(history.get("val_acc", []))
                tsl = _last(history.get("test_nll", []))
                tsa = _last(history.get("test_acc", []))
                log(f"train_loss(last): {tl}" if tl is not None else "train_loss(last): n/a")
                log(f"train_acc(last):  {ta}" if ta is not None else "train_acc(last): n/a")
                log(f"val_nll(last):   {vl}" if vl is not None else "val_nll(last): n/a")
                log(f"val_acc(last):   {va}" if va is not None else "val_acc(last): n/a")
                log(f"test_nll(last):  {tsl}" if tsl is not None else "test_nll(last): n/a")
                log(f"test_acc(last):  {tsa}" if tsa is not None else "test_acc(last): n/a")
            else:
                log("(no metrics/history found in checkpoint)")
            log("-----------------------------------")
        except Exception as e:
            log(f"[warn] Could not read base SNN checkpoint metrics: {e}")
    else:
        log("[info] No base SNN checkpoint provided; using random initialization for mod training.")

    # ----- build splits (unchanged) -----
    folds: List[Tuple[np.ndarray, Optional[np.ndarray], dict]] = []
    if external_val_available:
        tr_idx = np.arange(len(y_train))
        val_idx = None
        idx_path = run_dir / "Fold_1" / "split_indices.npz"
        save_indices(idx_path, tr_idx, val_idx, {"external_validation": True})
        folds = [(tr_idx, val_idx, {"source": "external_val", "path": str(idx_path)})]
        if k_folds and k_folds > 1:
            log("[info] External validation provided; ignoring k-fold configuration.")
        if fixed_split_path:
            log("[info] External validation provided; ignoring --fixed_split_path.")
    elif k_folds and k_folds > 1:
        if fixed_split_path:
            for k in range(1, k_folds+1):
                p = Path(fixed_split_path) if str(fixed_split_path).endswith(".npz") else Path(fixed_split_path)/f"fold_{k}_indices.npz"
                tr_idx, val_idx, meta = load_indices(p); folds.append((tr_idx, val_idx, {"source":"loaded","path":str(p)}))
            log(f"Loaded {k_folds} folds from {fixed_split_path}")
        else:
            for k,(tr_idx,val_idx) in enumerate(make_kfold_splits(y_train, k_folds, seed), start=1):
                idx_path = run_dir / f"Fold_{k}" / "fold_indices.npz"
                save_indices(idx_path, tr_idx, val_idx, {"k_folds":k_folds, "fold":k, "seed":seed})
                folds.append((tr_idx, val_idx, {"source":"created","path":str(idx_path)}))
            log(f"Created & saved {k_folds} stratified folds.")
    else:
        if use_validation_flag:
            if fixed_split_path:
                p = Path(fixed_split_path); tr_idx, val_idx, meta = load_indices(p)
                folds = [(tr_idx, val_idx, {"source":"loaded","path":str(p)})]; log(f"Loaded fixed split from {p}")
            else:
                tr_idx, val_idx = stratified_split_indices(y_train, val_fraction, seed)
                idx_path = run_dir / "Fold_1" / "split_indices.npz"
                save_indices(idx_path, tr_idx, val_idx, {"val_fraction":val_fraction, "seed":seed})
                folds = [(tr_idx, val_idx, {"source":"created","path":str(idx_path)})]; log(f"Saved split to {idx_path}")
        else:
            tr_idx = np.arange(len(y_train)); val_idx = None
            idx_path = run_dir / "Fold_1" / "split_indices.npz"
            save_indices(idx_path, tr_idx, val_idx, {"no_validation":True})
            folds = [(tr_idx, val_idx, {"source":"created","path":str(idx_path)})]; log("No validation: saved train indices.")

    manifest = []

    # ----- train per fold (unchanged control flow) -----
    for fold_id, (tr_idx, val_idx, meta) in enumerate(folds, start=1):
        multi_fold = (k_folds and k_folds > 1) and not external_val_available
        fold_dir = run_dir / (f"Fold_{fold_id}" if multi_fold else "Fold_1"); fold_dir.mkdir(parents=True, exist_ok=True)
        log(f"=== Fold {fold_id}/{len(folds)} ({meta['source']}) — indices: {meta['path']} ===")
        if external_val_available and y_val is not None:
            log(f"Train size: {len(tr_idx)} / External Val size: {len(y_val)}")
        elif val_idx is None:
            log(f"Train size (no validation): {len(tr_idx)}")
        else:
            log(f"Train/Val sizes: {len(tr_idx)} / {len(val_idx)}")

        state_fold = (
            {k: torch.nn.Parameter(v.detach().clone(), requires_grad=v.requires_grad) for k,v in state.items()}
            if reinit_per_fold else state
        )
        fold_mode = _select_mlp_mode(settings)
        fold_hidden_sizes = _resolve_mlp_hidden_sizes(settings)
        fold_mod_hidden_sizes = settings.get("mod_hidden_sizes")
        hidden_cfg = fold_mod_hidden_sizes if _is_snn_mode(fold_mode) else fold_hidden_sizes
        modulator_fold = build_modulator(
            settings,
            override_mode=fold_mode,
            hidden_sizes=hidden_cfg,
        )
        log(format_param_stats(state_fold, modulator_fold, prefix=f"Fold {fold_id}:", settings=settings))

        if settings.get("mlp_init_state_dict") is not None:
            modulator_fold.load_state_dict(settings["mlp_init_state_dict"])

        mod_optimizer = torch.optim.Adam(
            [{"params": modulator_fold.parameters(), "lr": settings["lr"]}],
            betas=(0.9, 0.999),
        )
        snn_optimizer = None
        snn_unlocked = False
        snn_trainables = _trainable_params_from(state_fold)
        if (not start_locked) and snn_trainables:
            snn_optimizer = torch.optim.Adam([{"params": snn_trainables, "lr": settings["lr"]}], betas=(0.9, 0.999))
            snn_unlocked = True

        loss_fn = nn.NLLLoss(); log_softmax = nn.LogSoftmax(dim=1)
        snn_mode_flag = _is_snn_mode(fold_mode)

        def _pack_params(): return {k: v.clone() for k, v in state_fold.items()}
        def _save_ckpt(path: Path, epoch: int, metrics: dict, history: dict):
            torch.save({
                "epoch": epoch,
                "metrics": metrics,
                "snn_params": _pack_params(),
                "mlp_state": modulator_fold.state_dict(),
                "history": history,
                "mlp_mode": settings.get("mlp_mode", DEFAULT_MLP_MODE),
                "ann_mode": settings.get("ann_mode", settings.get("mlp_mode", DEFAULT_MLP_MODE)),
                "mlp_interval": settings.get("mlp_interval"),
                "ann_interval": settings.get("ann_interval", settings.get("mlp_interval")),
                "mod_update_every_step": settings.get("mod_update_every_step", False),
                "mlp_hidden_sizes": settings.get("mlp_hidden_sizes"),
                "mlp_arch": settings.get("mlp_arch", DEFAULT_MLP_ARCH),
                "mod_hidden_sizes": settings.get("mod_hidden_sizes"),
                "snn_mod_hidden": settings.get("snn_mod_hidden"),
                "snn_mod_hidden_recurrent": settings.get("snn_mod_hidden_recurrent", False),
                "run_config": _config_snapshot(settings),
            }, path)

        history = {"train_loss": [], "val_nll": [], "val_acc": [], "train_acc": [], "test_nll": [], "test_acc": []}
        best_val_nll, best_val_acc, best_val_epoch = float("inf"), -1.0, -1
        best_ckpt_metric, best_ckpt_secondary, best_ckpt_epoch = float("-inf"), float("-inf"), -1
        val_data_x = x_val if external_val_available else x_train
        val_data_y = y_val if external_val_available else y_train
        val_indices = None if external_val_available else val_idx
        val_count = len(val_data_y) if (external_val_available and val_data_y is not None) else (len(val_idx) if val_idx is not None else 0)
        has_validation = use_validation_flag and (val_count > 0)
        epochs_no_improve = 0

        for e in range(settings["nb_epochs"]):
            local_loss, accs = [], []
            train_gen = sparse_data_generator_from_hdf5_spikes(
                x_train, y_train, settings["batch_size"], settings["nb_steps"],
                settings["nb_inputs"], settings["max_time"], shuffle=True, indices=tr_idx,
                augment_cfg=settings.get("augment_train"), postbin_time_mask=settings.get("postbin_mask_train", 0.0)
            )
            for x_local, y_local in train_gen:
                out, (mem_rec, spk_rec) = run_snn_modulated(
                    x_local.to_dense(), state_fold, settings, modulator_fold, settings["mlp_interval"],
                    training=True
                )
                m,_ = torch.max(out, 1)
                _, am = torch.max(m, 1)
                accs.append(torch.mean((y_local == am).float()).item())
                log_p = log_softmax(m); ground_loss = loss_fn(log_p, y_local)

                loss_val = ground_loss

                if settings.get("use_mod_reg", True):
                    nb_hidden = settings["nb_hidden"]; nb_outputs = settings["nb_outputs"]
                    N_samp, T = 8128, settings["nb_steps"]
                    sl, thetal = 1.0, 0.01; su, thetau = 0.06, 100.0
                    tmp  = (torch.clamp((1/T)*torch.sum(spk_rec, 1) - thetal, min=0.0))**2
                    L1   = torch.sum(tmp, (0,1))
                    tmp2 = (torch.clamp((1/nb_hidden)*torch.sum(spk_rec, (1,2)) - thetau, min=0.0))**2
                    L2   = torch.sum(tmp2)
                    reg_loss = (sl/(N_samp+nb_hidden+nb_outputs))*L1 + (su/N_samp)*L2
                    reg_loss = settings.get("mod_reg_scale", 1.0) * reg_loss
                    loss_val = loss_val + reg_loss

                if snn_mode_flag and settings.get("use_snn_mod_reg", False):
                    mod_spk_trace = getattr(modulator_fold, "_last_spk_rec", None)
                    if mod_spk_trace is not None:
                        mod_units = int(mod_spk_trace.size(-1))
                        N_samp, T = 8128, settings["nb_steps"]
                        sl, thetal = 1.0, 0.01; su, thetau = 0.06, 100.0
                        tmp  = (torch.clamp((1/T)*torch.sum(mod_spk_trace, 1) - thetal, min=0.0))**2
                        L1   = torch.sum(tmp, (0,1))
                        tmp2 = (torch.clamp((1/max(1, mod_units))*torch.sum(mod_spk_trace, (1,2)) - thetau, min=0.0))**2
                        L2   = torch.sum(tmp2)
                        mod_reg_loss = (sl/(N_samp+mod_units))*L1 + (su/N_samp)*L2
                        mod_reg_loss = settings.get("snn_mod_reg_scale", 1.0) * mod_reg_loss
                        loss_val = loss_val + mod_reg_loss
                mod_optimizer.zero_grad()
                if snn_optimizer is not None:
                    snn_optimizer.zero_grad()
                loss_val.backward()
                mod_optimizer.step()
                if snn_optimizer is not None:
                    snn_optimizer.step()
                if snn_mode_flag and isinstance(modulator_fold, SNNAdditiveModulator):
                    modulator_fold.clamp_mod_factors_()

                with torch.no_grad():
                    _clamp_state_params(state_fold, _resolve_param_ranges(settings))

                local_loss.append(float(loss_val.item()))

            train_loss = float(np.mean(local_loss)) if local_loss else float("nan")
            train_acc = float(np.mean(accs)) if accs else 0.0
            history["train_loss"].append(train_loss); history["train_acc"].append(train_acc)

            test_nll, test_acc = evaluate_testset_mod(state_fold, settings, x_test, y_test, modulator_fold, settings["mlp_interval"])
            history["test_nll"].append(test_nll); history["test_acc"].append(test_acc)

            if has_validation and val_data_x is not None and val_data_y is not None:
                val_nll, val_acc = eval_val_metrics_mod(
                    state_fold, settings, val_data_x, val_data_y,
                    indices=val_indices, modulator=modulator_fold, mlp_interval=settings["mlp_interval"]
                )
                history["val_nll"].append(val_nll); history["val_acc"].append(val_acc)
                log(f"[Fold {fold_id}] Epoch {e+1}: train_loss={train_loss:.5f} train_acc={train_acc:.5f} "
                    f"val_nll={val_nll:.5f} val_acc={val_acc:.5f} test_nll={test_nll:.5f} test_acc={test_acc:.5f}")

                metrics_payload = {"train_loss":train_loss,"train_acc":train_acc,"val_nll":val_nll,"val_acc":val_acc,
                                   "test_nll":test_nll,"test_acc":test_acc}
                if save_every_epoch:
                    _save_ckpt(fold_dir / f"mod_{e+1}.pth", e+1, metrics_payload, history)

                # unlock logic (unchanged)
                if (unlock_metric is not None) and (unlock_threshold is not None):
                    cond = (
                        (unlock_metric == "val_acc" and val_acc  >= unlock_threshold) or
                        (unlock_metric == "train_acc" and train_acc >= unlock_threshold)
                    )
                    if cond and not snn_unlocked:
                        log(f"Unlocking SNN ({unlock_metric} ≥ {unlock_threshold:.5f})")
                        if snn_trainables:
                            snn_optimizer = torch.optim.Adam([{"params": snn_trainables, "lr": settings["lr"]}], betas=(0.9, 0.999))
                        snn_unlocked = True

                improved = (val_nll < best_val_nll - 1e-6) or (abs(val_nll - best_val_nll) <= 1e-6 and val_acc > best_val_acc)
                if improved:
                    best_val_nll, best_val_acc, best_val_epoch = val_nll, val_acc, e+1
                    epochs_no_improve = 0
                    log(f"[Fold {fold_id}] New best (patience metric) at epoch {best_val_epoch} "
                        f"(val_nll={best_val_nll:.5f}, val_acc={best_val_acc:.5f})")
                else:
                    epochs_no_improve += 1
                    log(f"[Fold {fold_id}] No improvement ({epochs_no_improve}/{patience}) — best val_nll={best_val_nll:.5f} at epoch {best_val_epoch}")

                if epochs_no_improve >= patience:
                    log(f"[Fold {fold_id}] Early stopping."); break
            else:
                log(f"[Fold {fold_id}] Epoch {e+1}: train_loss={train_loss:.5f} train_acc={train_acc:.5f} "
                    f"(no validation) test_nll={test_nll:.5f} test_acc={test_acc:.5f}")
                metrics_payload = {"train_loss":train_loss,"train_acc":train_acc,"test_nll":test_nll,"test_acc":test_acc}
                if save_every_epoch:
                    _save_ckpt(fold_dir / f"mod_{e+1}.pth", e+1, metrics_payload, history)

            metric_value = val_acc if has_validation else train_acc
            better_metric = metric_value > best_ckpt_metric + 1e-6
            better_secondary = (abs(metric_value - best_ckpt_metric) <= 1e-6) and (test_acc > best_ckpt_secondary + 1e-6)
            if better_metric or better_secondary:
                best_ckpt_metric, best_ckpt_secondary, best_ckpt_epoch = metric_value, test_acc, e+1
                _save_ckpt(fold_dir / "mod_best.pth", best_ckpt_epoch, metrics_payload, history)

        manifest.append({"fold": fold_id, "indices_path": meta["path"], "best_epoch": best_ckpt_epoch})

        best_mod_ckpt = fold_dir / "mod_best.pth"
        if best_mod_ckpt.exists():
            try:
                state_eval, mlp_eval = load_modulated_checkpoint(best_mod_ckpt, settings, quiet=True)
                spike_stats = compute_mod_test_spike_stats(
                    state_eval,
                    settings,
                    mlp_eval,
                    settings["mlp_interval"],
                    x_test,
                    y_test,
                )
                log(f"[Fold {fold_id}] Test spike stats (Mod) — {_format_spike_stats(spike_stats)}")
            except Exception as exc:
                log(f"[Fold {fold_id}] Spike stats computation failed (Mod): {exc}")
        else:
            log(f"[Fold {fold_id}] Spike stats (Mod) skipped: {best_mod_ckpt} not found.")

    log_file.close()
    return {"fold_summaries": manifest, "run_dir": str(run_dir)}

# ===== Block 2: Adjustables + Run (CLI) =====
import argparse



__all__ = [name for name in globals() if not name.startswith('__')]
