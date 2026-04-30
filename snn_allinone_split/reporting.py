"""Run config logging and parameter/stat summaries."""
from .grouping import *

def _snn_shape_str(state: Optional[Dict[str, torch.nn.Parameter]]) -> Optional[str]:
    if not state:
        return None
    w1 = state.get("w1")
    w2 = state.get("w2")
    if w1 is None or w2 is None:
        return None
    if not hasattr(w1, "shape") or not hasattr(w2, "shape"):
        return None
    if w1.dim() < 2 or w2.dim() < 2:
        return None
    try:
        in_dim, hidden_dim = int(w1.shape[0]), int(w1.shape[1])
        out_dim = int(w2.shape[1])
    except Exception:
        return None
    return f"{in_dim}-{hidden_dim}-{out_dim}"


def _mlp_shape_str(mlp: Optional[nn.Module]) -> Optional[str]:
    if mlp is None:
        return None
    arch = getattr(mlp, "arch", None)
    if arch in {"rnn", "lstm"}:
        input_size = getattr(mlp, "input_size", None)
        output_size = getattr(mlp, "output_size", None)
        hidden_size = getattr(mlp, "rnn_hidden_size", None)
        num_layers = getattr(mlp, "rnn_num_layers", None)
        if None not in (input_size, output_size, hidden_size, num_layers):
            arch_name = arch.upper()
            return f"{arch_name}({int(input_size)}->{int(hidden_size)}x{int(num_layers)}->{int(output_size)})"
    linear_layers: List[nn.Module] = list(getattr(mlp, "_linear_layers", []) or [])
    if not linear_layers:
        linear_layers = [mod for mod in mlp.modules() if isinstance(mod, nn.Linear)]
    if not linear_layers:
        spiking_layers = getattr(mlp, "layers", None)
        if spiking_layers:
            linear_layers = [layer for layer in spiking_layers if hasattr(layer, "out_dim")]
    if not linear_layers:
        return None
    input_size = getattr(mlp, "input_size", None)
    if input_size is None:
        first = linear_layers[0]
        input_size = getattr(first, "in_features", None)
        if input_size is None:
            input_size = getattr(first, "in_dim", None)
    if input_size is None:
        return None
    dims: List[int] = [int(input_size)]
    for layer in linear_layers:
        out_features = getattr(layer, "out_features", None)
        if out_features is None:
            out_features = getattr(layer, "out_dim", None)
        if out_features is None:
            return None
        dims.append(int(out_features))
    return "-".join(str(d) for d in dims)


def _nm_shape_str(modulator) -> Optional[str]:
    mapper = getattr(modulator, "nm_mapper", None)
    if mapper is None:
        return None
    parts = []
    if getattr(mapper, "hidden_mlp", None) is not None and mapper.hidden_per_neuron > 0 and mapper.hidden_targets > 0:
        h_shape = _mlp_shape_str(mapper.hidden_mlp) or f"{mapper.hidden_per_neuron}->{mapper.hidden_width}->5"
        parts.append(f"H:{h_shape} (targets={mapper.hidden_targets})")
    if getattr(mapper, "output_mlp", None) is not None and mapper.output_per_neuron > 0 and mapper.output_targets > 0:
        o_shape = _mlp_shape_str(mapper.output_mlp) or f"{mapper.output_per_neuron}->{mapper.output_width}->2"
        parts.append(f"O:{o_shape} (targets={mapper.output_targets})")
    return " | ".join(parts) if parts else None


def _slice_len(slc: Optional[slice]) -> int:
    if slc is None:
        return 0
    try:
        return int(slc.stop) - int(slc.start)
    except Exception:
        return 0


def _ordered_slice_sizes(slices: Optional[Dict[str, slice]], order: List[str]) -> List[Tuple[str, int]]:
    if not slices:
        return []
    items: List[Tuple[str, int]] = []
    for name in order:
        slc = slices.get(name)
        if slc is None:
            continue
        size = _slice_len(slc)
        items.append((name, size))
    return items


def _count_modulated_neurons(
    layout: Optional['GroupLayout'],
    union_idx: Optional[torch.Tensor],
    target_count: int
) -> Optional[int]:
    if union_idx is None:
        return None
    idx = union_idx.detach().cpu()
    if layout is None or not getattr(layout, "enabled", False) or getattr(layout, "forward", None) is None:
        return int(idx.numel())
    if idx.numel() == 0:
        return 0
    forward = layout.forward
    try:
        selected = forward.index_select(0, idx)
    except Exception:
        return int(idx.numel())
    mask = selected.sum(dim=0) > 0
    return int(mask.sum().item())


def format_param_stats(
    state: Optional[Dict[str, torch.nn.Parameter]],
    modulator: Optional[nn.Module] = None,
    prefix: str = "",
    settings: Optional[Dict[str, Any]] = None,
) -> str:
    snn_trainable, snn_frozen = _count_snn_params(state)
    snn_total = snn_trainable + snn_frozen
    lines = []
    lines.append("[ParamStats] " + (f"{prefix.strip()} " if prefix else "") + "SNN trainable: {0:,} | SNN frozen: {1:,} | SNN total: {2:,}".format(
        snn_trainable, snn_frozen, snn_total
    ))
    snn_shape = _snn_shape_str(state)
    if snn_shape:
        lines.append("[ParamStats] " + (f"{prefix.strip()} " if prefix else "") + f"SNN shape: {snn_shape}")
    if modulator is not None:
        mod_trainable, mod_frozen = _count_module_params(modulator)
        mod_total = mod_trainable + mod_frozen
        mod_line = "Modulator trainable: {0:,} | Modulator frozen: {1:,} | Modulator total: {2:,}".format(
            mod_trainable, mod_frozen, mod_total
        )
        if snn_total > 0 and mod_total > 0:
            mod_pct = 100.0 * mod_total / snn_total
            mod_line += f" | Modulator vs SNN: {mod_pct:.2f}%"
        arch = getattr(modulator, "arch", None)
        if arch:
            mod_line += f" | Arch: {arch.upper()}"
        lines.append("[ParamStats] " + (f"{prefix.strip()} " if prefix else "") + mod_line)

        mod_shape = None
        if hasattr(modulator, "_linear_layers") or hasattr(modulator, "layers"):
            mod_shape = _mlp_shape_str(modulator)
        if mod_shape:
            lines.append("[ParamStats] " + (f"{prefix.strip()} " if prefix else "") + f"Modulator shape: {mod_shape}")
            in_slices = getattr(modulator, "in_slices", None)
            in_items = _ordered_slice_sizes(
                in_slices,
                ["alpha_1", "beta_1", "thr", "reset", "rest", "alpha_2", "beta_2", "in_flat", "hid_flat", "out_flat"],
            )
            if in_items:
                in_total = sum(size for _, size in in_items)
                parts = ", ".join(f"{name}={size}" for name, size in in_items)
                lines.append("[ParamStats] " + (f"{prefix.strip()} " if prefix else "") + f"Modulator input: {parts} (total={in_total})")
                if settings is not None:
                    hid_flat_size = 0
                    for name, size in in_items:
                        if name == "hid_flat":
                            hid_flat_size = size
                            break
                    hid_flat_enabled = bool(settings.get("mlp_in_mask", {}).get("hid_flat", True))
                    mod_hid_flat_group = bool(settings.get("mod_hid_flat_group", False))
                    mod_hid_flat_mod_only = bool(settings.get("mod_hid_flat_modulated_only", False))
                    lines.append(
                        "[ParamStats] "
                        + (f"{prefix.strip()} " if prefix else "")
                        + f"Hid_flat dim: {hid_flat_size} (enabled={hid_flat_enabled}, group={mod_hid_flat_group}, mod_only={mod_hid_flat_mod_only})"
                    )

            out_items: List[Tuple[str, int]] = []
            if getattr(modulator, "use_neuromodulators", False) and getattr(modulator, "nm_out_slices", None):
                nm_slices = getattr(modulator, "nm_out_slices", {})
                out_items = [(f"nm_{name}", size) for name, size in _ordered_slice_sizes(nm_slices, ["hidden", "output"])]
            else:
                out_slices = getattr(modulator, "out_slices", None)
                out_items = _ordered_slice_sizes(
                    out_slices,
                    ["alpha_1", "beta_1", "thr", "reset", "rest", "alpha_2", "beta_2"],
                )
            if out_items:
                out_total = sum(size for _, size in out_items)
                parts = ", ".join(f"{name}={size}" for name, size in out_items)
                lines.append("[ParamStats] " + (f"{prefix.strip()} " if prefix else "") + f"Modulator output: {parts} (total={out_total})")

            if settings is not None:
                mod_mask = settings.get("_mod_fixed_mask")
                if mod_mask is not None:
                    nb_hidden = int(settings.get("nb_hidden", 0))
                    nb_outputs = int(settings.get("nb_outputs", 0))
                    hidden_layout = _get_group_layout(settings, 0, nb_hidden)
                    output_layout = _get_group_layout(settings, 1, nb_outputs)
                    hidden_count = _count_modulated_neurons(hidden_layout, mod_mask.get("hidden_union_idx"), nb_hidden)
                    output_count = _count_modulated_neurons(output_layout, mod_mask.get("output_union_idx"), nb_outputs)
                    if hidden_count is not None and output_count is not None:
                        lines.append(
                            "[ParamStats] " + (f"{prefix.strip()} " if prefix else "")
                            + f"Modulated neurons: hidden={hidden_count}/{nb_hidden}, output={output_count}/{nb_outputs}"
                        )
        nm_shape = _nm_shape_str(modulator)
        if nm_shape:
            lines.append("[ParamStats] " + (f"{prefix.strip()} " if prefix else "") + f"NM MLP: {nm_shape}")
    return "\n".join(lines)

# -------------------------
# Run header logging
# -------------------------
def _to_builtin(v):
    import numpy as _np
    if isinstance(v, _np.generic): return v.item()
    if isinstance(v, Path): return str(v)
    if isinstance(v, (list, tuple)): return type(v)(_to_builtin(x) for x in v)
    if isinstance(v, dict): return {str(k): _to_builtin(vv) for k, vv in v.items()}
    return v

def _config_value(v):
    import numpy as _np
    if isinstance(v, _np.generic):
        return v.item()
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, torch.Tensor):
        return {"tensor_shape": list(v.shape), "dtype": str(v.dtype)}
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_config_value(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _config_value(vv) for k, vv in v.items()}
    return str(v)

def _config_snapshot(settings: Dict[str, Any]) -> Dict[str, Any]:
    skip_keys = {"mlp_init_state_dict"}
    snapshot = {}
    for key, value in settings.items():
        if key in skip_keys or str(key).startswith("_"):
            continue
        snapshot[str(key)] = _config_value(value)
    return snapshot

def write_run_config(run_dir: Path, settings: Dict[str, Any], extras: Dict[str, Any]) -> Path:
    config_path = run_dir / "run_config.json"
    payload = {
        "settings": _config_snapshot(settings),
        "controls": _config_value(extras or {}),
    }
    with config_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return config_path

def _section(title, d):
    lines = [f"== {title} =="]
    for k in sorted(d.keys()):
        lines.append(f"{k}: {_to_builtin(d[k])}")
    return "\n".join(lines)

def log_run_header(log, run_dir, settings, **kw):
    # masks as disabled lists for readability
    in_mask  = settings.get("mlp_in_mask", {})
    out_mask = settings.get("mlp_out_mask", {})
    in_disabled  = [k for k,v in in_mask.items()  if v is False]
    out_disabled = [k for k,v in out_mask.items() if v is False]

    model = dict(
        nb_inputs=settings["nb_inputs"], nb_hidden=settings["nb_hidden"], nb_outputs=settings["nb_outputs"]
    )
    sim = dict(time_step=settings["time_step"], nb_steps=settings["nb_steps"], max_time=settings["max_time"])
    sim["input_delay_steps"] = settings.get("input_delay_steps")
    sim["use_input_delay"] = settings.get("use_input_delay")
    sim["input_delay_temp"] = settings.get("input_delay_temp")
    data = dict(
        batch_size=settings["batch_size"],
        cache_dir=settings["cache_dir"], cache_subdir=settings["cache_subdir"],
        train_file=settings["train_file"], test_file=settings["test_file"],
        val_file=settings.get("val_file"),
        nb_inputs_raw=settings.get("nb_inputs_raw"),
        channel_compress_enable=settings.get("channel_compress_enable", False),
        channel_compress_factor=settings.get("channel_compress_factor", 1),
        channel_compress_target=settings.get("channel_compress_target"),
        channel_compress_mode=settings.get("channel_compress_mode"),
        channel_compress_factor_snn=settings.get("channel_compress_factor_snn"),
        channel_compress_factor_mod=settings.get("channel_compress_factor_mod"),
        nb_inputs_mod=settings.get("nb_inputs_mod"),
        channel_compress_mlp_hidden_sizes=settings.get("channel_compress_mlp_hidden_sizes"),
    )
    train = dict(
        lr=settings["lr"], nb_epochs=settings["nb_epochs"],
        tau_syn=settings.get("tau_syn"), tau_mem=settings.get("tau_mem"),
        tau_match_clip=settings.get("tau_match_clip"),
        tau_decay_dt=_effective_tau_decay_dt(settings),
        tau_scale=settings.get("paper_tau_scale", 1.0),
        weight_scale=settings.get("weight_scale"),
        hidden_dropout_p=settings.get("hidden_dropout_p", 0.0),
    )

    aug_train = dict(**(settings.get("augment_train", {}) or {}), postbin_mask=settings.get("postbin_mask_train", 0.0))
    aug_eval  = dict(**(settings.get("augment_eval",  {}) or {}), postbin_mask=settings.get("postbin_mask_eval",  0.0))
    aug_test  = dict(**(settings.get("augment_test",  {}) or {}), postbin_mask=settings.get("postbin_mask_test",  0.0))

    nm_cfg = settings.get("nm_cfg", {}) or {}
    nm_cfg_structured = {
        "enable": bool(nm_cfg.get("enable", False)),
        "hidden_per_neuron": int(nm_cfg.get("hidden_per_neuron", 0)),
        "output_per_neuron": int(nm_cfg.get("output_per_neuron", 0)),
        "init_scale": nm_cfg.get("init_scale"),
        "activation_kind": nm_cfg.get("activation_kind"),
        "flat_order": nm_cfg.get("flat_order"),
        "warm_start": bool(nm_cfg.get("warm_start", False)),
        "init_identity": bool(nm_cfg.get("init_identity", False)),
        "neuron_fraction_enable": bool(nm_cfg.get("neuron_fraction_enable", False)),
        "neuron_fraction": nm_cfg.get("neuron_fraction"),
        "param_fraction_enable": bool(nm_cfg.get("param_fraction_enable", False)),
        "param_fraction": nm_cfg.get("param_fraction"),
        "mapper_type": nm_cfg.get("mapper_type"),
        "hidden_activation": nm_cfg.get("hidden_activation"),
        "hidden_mlp_hidden": nm_cfg.get("hidden_mlp_hidden"),
        "output_mlp_hidden": nm_cfg.get("output_mlp_hidden"),
        "mapper_hidden_exact": nm_cfg.get("mapper_hidden_exact"),
    }

    modulator = dict(
        mlp_hidden_sizes=settings.get("mlp_hidden_sizes"),
        mod_hidden_sizes=settings.get("mod_hidden_sizes"),
        mlp_interval=settings.get("mlp_interval"),
        mod_update_every_step=settings.get("mod_update_every_step"),
        mlp_mode=settings.get("mlp_mode", "mlp_sub"),
        mlp_in_disable=in_disabled, mlp_out_disable=out_disabled,
        snn_mod_hidden=settings.get("snn_mod_hidden"),
        snn_mod_gain_init=settings.get("snn_mod_gain_init"),
        snn_mod_weight_scale=settings.get("snn_mod_weight_scale"),
        snn_mod_hidden_recurrent=settings.get("snn_mod_hidden_recurrent"),
        snn_sub_scale_init=settings.get("snn_sub_scale_init"),
        snn_sub_bias_init=settings.get("snn_sub_bias_init"),
        ann_output_activation=settings.get("ann_output_activation_effective") or settings.get("ann_output_activation"),
        ann_output_activation_override=settings.get("ann_output_activation"),
        group_cfg=settings.get("group_cfg"),
        mod_fixed_mask_enable=(settings.get("mod_mask_cfg", {}) or {}).get("fixed_enable"),
        mod_fixed_mask_seed=(settings.get("mod_mask_cfg", {}) or {}).get("fixed_seed"),
        mod_fixed_mask_flat_inputs=(settings.get("mod_mask_cfg", {}) or {}).get("fixed_flat_inputs"),
        mod_fixed_mask_zero_fallback=(settings.get("mod_mask_cfg", {}) or {}).get("zero_fallback"),
        mod_hid_flat_group=settings.get("mod_hid_flat_group"),
        mod_hid_flat_modulated_only=settings.get("mod_hid_flat_modulated_only"),
    )

    misc = dict(save_dir=settings.get("save_dir"))
    extras = dict(**(kw or {}))
    config_path = write_run_config(run_dir, settings, extras)

    log("\n===== RUN CONFIG =====")
    log(f"[info] Wrote machine-readable run config: {config_path}")
    log(_section("Run Dir", {"path": str(run_dir)}))
    log(_section("Model", model))
    log(_section("Simulation", sim))
    log(_section("Data", data))
    log(_section("Training", train))
    log(_section("Augment (train)", aug_train))
    log(_section("Augment (eval)",  aug_eval))
    log(_section("Augment (test)",  aug_test))
    if any(v is not None for v in modulator.values()):
        log(_section("Modulator", modulator))
    if any(v is not None for v in nm_cfg_structured.values()):
        log(_section("Neuromodulation", nm_cfg_structured))
    ranges = settings.get("param_ranges")
    if ranges:
        log(_section("Param Ranges", ranges))
    if "param_timescales" in settings:
        log(_section("Param Timescales", settings.get("param_timescales", {})))
    if "param_smoothing" in settings:
        log(_section("Param Smoothing", settings.get("param_smoothing", {})))
    log(_section("Controls", extras))
    if settings.get("mod_hid_flat_modulated_only") and not (settings.get("mod_mask_cfg", {}) or {}).get("fixed_enable"):
        log("[warn] mod_hid_flat_modulated_only is true but mod_fixed_mask_enable is false; hid_flat will not be filtered.")
    mod_output_size = settings.get("_mod_output_size")
    if isinstance(mod_output_size, int) and mod_output_size == 0:
        log("[warn] Modulator output size is 0; no parameters will be modulated. Training will still run.")
    log("=" * 24 + "\n")

# -------------------------
# Augment helpers (pre/post bin)
# -------------------------


__all__ = [name for name in globals() if not name.startswith('__')]
