# =============================================================================
# SNN ALL-IN-ONE SMALL CLI
#
# Command-line runner for snn_allinone_small.py. The model/data/training code
# lives in snn_allinone_small.py so readers can skip this file unless they want
# to run experiments from the terminal.
# =============================================================================

import argparse
from typing import Optional, Sequence

from snn_allinone_small import *

# =============================================================================
# SECTION 1: Reporting, Run Config, and Logging
# =============================================================================

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

# =============================================================================
# SECTION 2: Base SNN Evaluation and Training
# =============================================================================
@torch.no_grad()
def eval_val_metrics(state, settings, x_data, y_data, indices: Optional[np.ndarray] = None):
    loss_fn = nn.NLLLoss(reduction="mean")
    log_softmax_fn = nn.LogSoftmax(dim=1)
    nlls, accs = [], []
    gen = sparse_data_generator_from_hdf5_spikes(
        x_data, y_data, settings["batch_size"], settings["nb_steps"],
        settings["nb_inputs"], settings["max_time"], shuffle=False, indices=indices,
        augment_cfg=settings.get("augment_eval"), postbin_time_mask=settings.get("postbin_mask_eval", 0.0)
    )
    for x_local, y_local in gen:
        output, _ = run_snn_hetero(x_local.to_dense(), state, settings)
        m, _ = torch.max(output, 1)
        log_p_y = log_softmax_fn(m)
        nlls.append(loss_fn(log_p_y, y_local).item())
        _, am = torch.max(m, 1)
        accs.append(torch.mean((y_local == am).float()).item())
    return float(np.mean(nlls)) if nlls else float("nan"), float(np.mean(accs)) if accs else 0.0

@torch.no_grad()
def evaluate_testset(state, settings, x_test, y_test):
    loss_fn = nn.NLLLoss(reduction="mean")
    log_softmax_fn = nn.LogSoftmax(dim=1)
    nlls, accs = [], []
    gen = sparse_data_generator_from_hdf5_spikes(
        x_test, y_test, settings["batch_size"], settings["nb_steps"],
        settings["nb_inputs"], settings["max_time"], shuffle=False,
        augment_cfg=settings.get("augment_test"),
        postbin_time_mask=settings.get("postbin_mask_test", 0.0)
    )
    for x_local, y_local in gen:
        output, _ = run_snn_hetero(x_local.to_dense(), state, settings)
        m, _ = torch.max(output, 1)
        log_p_y = log_softmax_fn(m)
        nlls.append(loss_fn(log_p_y, y_local).item())
        _, am = torch.max(m, 1)
        accs.append(torch.mean((y_local == am).float()).item())
    test_nll = float(np.mean(nlls)) if nlls else float("nan")
    test_acc = float(np.mean(accs)) if accs else 0.0
    print(f"Final Test — NLL: {test_nll:.5f}, Accuracy: {test_acc:.5f}")
    return test_nll, test_acc

# -------------------------
# Train SNN (returns best ckpt path too)
# -------------------------
def train_snn_hetero(
    state: dict, settings: dict,
    x_train, y_train, x_test, y_test,
    x_val=None, y_val=None,
    save_every_epoch: bool = True,
    use_validation: bool = True,
    val_fraction: float = 0.1,
    k_folds: int = 1,
    patience: int = 20,
    seed: int = 0,
    fixed_split_path: str = None,
    reinit_per_fold: bool = True,
    chosen_metric: str = "val_acc",                      # "val_acc" or "train_acc"
    chosen_threshold: Optional[float] = None,            # threshold for chosen metric
    run_index: Optional[int] = None,
):
    run_dir = ensure_run_dir(settings["save_dir"], run_index=run_index)
    print(f"Saving to: {run_dir}")
    log_path = run_dir / "training_log.txt"
    log_file = open(log_path, "a", buffering=1)
    print(f"Logging to: {log_path}")
    def log(msg: str):
        print(msg); print(msg, file=log_file)

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
        mode="snn",
        use_validation=use_validation_flag,
        val_fraction=val_fraction,
        k_folds=k_folds,
        patience=patience,
        seed=seed,
        fixed_split_path=fixed_split_path,
        reinit_per_fold=reinit_per_fold,
        chosen_metric=chosen_metric,
        chosen_threshold=chosen_threshold,
        snn_reg_enabled=settings.get("use_snn_reg"),
        snn_train_disabled=snn_disabled,
        external_val=external_val_available,
        external_val_file=val_dataset_name,
    )

    # ----- build splits (unchanged) -----
    folds = []
    if external_val_available:
        tr_idx = np.arange(len(y_train))
        val_idx = None
        idx_path = Path(run_dir) / "Fold_1" / "split_indices.npz"
        save_indices(idx_path, tr_idx, val_idx, meta={"external_validation": True})
        folds = [(tr_idx, val_idx, {"source": "external_val", "path": str(idx_path)})]
        if k_folds and k_folds > 1:
            log("[info] External validation provided; ignoring k-fold configuration.")
        if fixed_split_path:
            log("[info] External validation provided; ignoring --fixed_split_path.")
    elif k_folds and k_folds > 1:
        if fixed_split_path:
            for k in range(1, k_folds + 1):
                p = Path(fixed_split_path) if str(fixed_split_path).endswith(".npz") else Path(fixed_split_path) / f"fold_{k}_indices.npz"
                tr_idx, val_idx, meta = load_indices(p)
                folds.append((tr_idx, val_idx, {"source": "loaded", "path": str(p)}))
            log(f"Loaded {k_folds} fixed folds from {fixed_split_path}")
        else:
            kfold_pairs = make_kfold_splits(y_train, k_folds=k_folds, seed=seed)
            for k, (tr_idx, val_idx) in enumerate(kfold_pairs, start=1):
                idx_path = (Path(run_dir) / f"Fold_{k}" / "fold_indices.npz")
                save_indices(idx_path, tr_idx, val_idx, meta={"k_folds": k_folds, "fold": k, "seed": seed})
                folds.append((tr_idx, val_idx, {"source": "created", "path": str(idx_path)}))
            log(f"Created & saved {k_folds} stratified folds.")
    else:
        if use_validation_flag:
            if fixed_split_path:
                p = Path(fixed_split_path)
                tr_idx, val_idx, meta = load_indices(p)
                folds = [(tr_idx, val_idx, {"source": "loaded", "path": str(p)})]
                log(f"Loaded fixed train/val split from {p}")
            else:
                tr_idx, val_idx = stratified_split_indices(y_train, val_fraction=val_fraction, seed=seed)
                idx_path = Path(run_dir) / "Fold_1" / "split_indices.npz"
                save_indices(idx_path, tr_idx, val_idx, meta={"val_fraction": val_fraction, "seed": seed})
                folds = [(tr_idx, val_idx, {"source": "created", "path": str(idx_path)})]
                log(f"Created & saved train/val split to {idx_path}")
        else:
            tr_idx = np.arange(len(y_train))
            val_idx = None
            idx_path = Path(run_dir) / "Fold_1" / "split_indices.npz"
            save_indices(idx_path, tr_idx, val_idx, meta={"no_validation": True})
            folds = [(tr_idx, val_idx, {"source": "created", "path": str(idx_path)})]
            log("No validation: saved train indices for reproducibility.")

    best_ckpt_paths = []
    index_manifest = []

    # ----- train per fold (unchanged control flow) -----
    for fold_id, (tr_idx, val_idx, idx_meta) in enumerate(folds, start=1):
        multi_fold = (k_folds and k_folds > 1) and not external_val_available
        fold_dir = run_dir / (f"Fold_{fold_id}" if multi_fold else "Fold_1")
        fold_dir.mkdir(parents=True, exist_ok=True)
        log(f"=== Fold {fold_id}/{len(folds)} ({idx_meta['source']}) — indices: {idx_meta['path']} ===")
        if external_val_available and y_val is not None:
            log(f"Train size: {len(tr_idx)} / External Val size: {len(y_val)}")
        elif val_idx is None:
            log(f"Train size (no validation): {len(tr_idx)}")
        else:
            log(f"Train/Val sizes: {len(tr_idx)} / {len(val_idx)}")

        state_fold = setup_model(settings) if reinit_per_fold else state
        log(format_param_stats(state_fold, prefix=f"Fold {fold_id}:", settings=settings))
        snn_trainable_params = _trainable_params_from(state_fold)
        optimizer = torch.optim.Adam(snn_trainable_params, lr=settings["lr"], betas=(0.9, 0.999)) if snn_trainable_params else None
        loss_fn = nn.NLLLoss(); log_softmax_fn = nn.LogSoftmax(dim=1)
        if optimizer is None:
            log("[info] All SNN parameters frozen for this fold; training will run without optimizer updates.")

        def _make_params_dict():
            return {k: v.clone() for k, v in state_fold.items()}
        def _save_checkpoint(path: Path, epoch: int, metrics: dict, hist: dict):
            torch.save({
                'epoch': epoch,
                'metrics': metrics,
                'params': _make_params_dict(),
                'history': hist,
                'run_config': _config_snapshot(settings),
            }, path)

        history = {"train_loss": [], "val_nll": [], "val_acc": [], "train_acc": [], "test_nll": [], "test_acc": []}
        best_val_nll, best_val_acc, best_val_epoch = float("inf"), -1.0, -1
        best_ckpt_metric, best_ckpt_secondary, best_ckpt_epoch = float("-inf"), float("-inf"), -1
        epochs_no_improve = 0; chosen_saved = False
        warned_no_val_for_chosen = False
        val_data_x = x_val if external_val_available else x_train
        val_data_y = y_val if external_val_available else y_train
        val_indices = None if external_val_available else val_idx
        val_count = len(val_data_y) if (external_val_available and val_data_y is not None) else (len(val_idx) if val_idx is not None else 0)
        has_validation = use_validation_flag and (val_count > 0)

        settings_train = dict(settings)
        settings_train["training"] = True
        for e in range(settings["nb_epochs"]):
            local_loss, accs = [], []
            train_gen = sparse_data_generator_from_hdf5_spikes(
                x_train, y_train, settings["batch_size"], settings["nb_steps"],
                settings["nb_inputs"], settings["max_time"], shuffle=True, indices=tr_idx,
                augment_cfg=settings.get("augment_train"), postbin_time_mask=settings.get("postbin_mask_train", 0.0)
            )
            for x_local, y_local in train_gen:
                output, recs = run_snn_hetero(x_local.to_dense(), state_fold, settings_train)
                _, spks = recs
                m, _ = torch.max(output, 1)
                _, am = torch.max(m, 1)
                accs.append(torch.mean((y_local == am).float()).item())

                log_p_y = log_softmax_fn(m)
                ground_loss = loss_fn(log_p_y, y_local)

                if settings.get("use_snn_reg", True):
                    nb_hidden = settings["nb_hidden"]; nb_outputs = settings["nb_outputs"]
                    N_samp, T = 8128, settings["nb_steps"]
                    sl, thetal = 1.0, 0.01; su, thetau = 0.06, 100.0
                    tmp  = (torch.clamp((1/T) * torch.sum(spks, 1) - thetal, min=0.0)) ** 2
                    L1   = torch.sum(tmp, (0,1))
                    tmp2 = (torch.clamp((1/nb_hidden) * torch.sum(spks, (1,2)) - thetau, min=0.0)) ** 2
                    L2   = torch.sum(tmp2)
                    reg_loss = (sl/(N_samp + nb_hidden + nb_outputs)) * L1 + (su/N_samp) * L2
                    reg_loss = settings.get("snn_reg_scale", 1.0) * reg_loss
                    loss_val = ground_loss + reg_loss
                else:
                    loss_val = ground_loss
                if optimizer is not None:
                    optimizer.zero_grad()
                    loss_val.backward()
                    optimizer.step()
                elif loss_val.requires_grad:
                    loss_val.backward()

                with torch.no_grad():
                    _clamp_state_params(state_fold, _resolve_param_ranges(settings))

                local_loss.append(float(loss_val.item()))

            train_loss = float(np.mean(local_loss)) if local_loss else float("nan")
            train_acc = float(np.mean(accs)) if accs else 0.0
            history["train_loss"].append(train_loss); history["train_acc"].append(train_acc)

            test_nll, test_acc = evaluate_testset(state_fold, settings, x_test, y_test)
            history["test_nll"].append(test_nll); history["test_acc"].append(test_acc)

            if has_validation and val_data_x is not None and val_data_y is not None:
                val_nll, val_acc = eval_val_metrics(
                    state_fold, settings, val_data_x, val_data_y, indices=val_indices
                )
                history["val_nll"].append(val_nll); history["val_acc"].append(val_acc)
                log(f"[Fold {fold_id}] Epoch {e+1}: train_loss={train_loss:.5f} train_acc={train_acc:.5f} "
                    f"val_nll={val_nll:.5f} val_acc={val_acc:.5f} test_nll={test_nll:.5f} test_acc={test_acc:.5f}")

                metrics_payload = {"train_loss":train_loss,"train_acc":train_acc,"val_nll":val_nll,"val_acc":val_acc,
                                   "test_nll":test_nll,"test_acc":test_acc}
                if save_every_epoch:
                    _save_checkpoint(fold_dir / f"snn_{e+1}.pth", e+1, metrics_payload, history)

                # 'chosen' checkpoint logic (unchanged)
                if (chosen_threshold is not None) and (not chosen_saved):
                    metric_value = val_acc if chosen_metric == "val_acc" else train_acc if chosen_metric == "train_acc" else None
                    if (metric_value is not None) and (metric_value >= chosen_threshold):
                        _save_checkpoint(fold_dir / "chosen.pth", e+1, metrics_payload, history)
                        log(f"[Fold {fold_id}] Saved 'chosen' ({chosen_metric} {metric_value:.5f} ≥ {chosen_threshold:.5f})")
                        chosen_saved = True

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
                    _save_checkpoint(fold_dir / f"snn_{e+1}.pth", e+1, metrics_payload, history)

                if (chosen_threshold is not None) and (not chosen_saved):
                    if chosen_metric == "val_acc":
                        if not warned_no_val_for_chosen:
                            log("[info] chosen_metric=val_acc but validation is disabled; 'chosen' cannot be saved on val_acc.")
                            warned_no_val_for_chosen = True
                    elif chosen_metric == "train_acc" and train_acc >= chosen_threshold:
                        _save_checkpoint(fold_dir / "chosen.pth", e+1,
                                         {"train_loss":train_loss,"train_acc":train_acc,
                                          "test_nll":test_nll,"test_acc":test_acc}, history)
                        log(f"[Fold {fold_id}] Saved 'chosen' (train_acc {train_acc:.5f} ≥ {chosen_threshold:.5f})")
                        chosen_saved = True

                if train_loss < best_val_nll:
                    best_val_nll, best_val_epoch = train_loss, e+1

            metric_value = val_acc if has_validation else train_acc
            better_metric = metric_value > best_ckpt_metric + 1e-6
            better_secondary = (abs(metric_value - best_ckpt_metric) <= 1e-6) and (test_acc > best_ckpt_secondary + 1e-6)
            if better_metric or better_secondary:
                best_ckpt_metric, best_ckpt_secondary, best_ckpt_epoch = metric_value, test_acc, e+1
                _save_checkpoint(fold_dir / "snn_best.pth", best_ckpt_epoch, metrics_payload, history)

        index_manifest.append({"fold": fold_id, "indices_path": idx_meta["path"], "best_epoch": best_ckpt_epoch})
        best_ckpt_paths.append(str(fold_dir / "snn_best.pth"))

        best_ckpt_path = fold_dir / "snn_best.pth"
        if best_ckpt_path.exists():
            try:
                state_eval = load_base_snn_state(best_ckpt_path, settings.get("snn_train_flags"), settings=settings)
                spike_stats = compute_snn_test_spike_stats(state_eval, settings, x_test, y_test)
                log(f"[Fold {fold_id}] Test spike stats — {_format_spike_stats(spike_stats)}")
            except Exception as exc:
                log(f"[Fold {fold_id}] Spike stats computation failed: {exc}")
        else:
            log(f"[Fold {fold_id}] Spike stats skipped: {best_ckpt_path} not found.")

    log_file.close()
    return {"fold_summaries": index_manifest, "run_dir": str(run_dir), "best_ckpt_paths": best_ckpt_paths}

# =============================================================================
# SECTION 3: Checkpoints and Stockpile Loading
# =============================================================================
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
    flags = _normalize_train_flags(train_flags)
    return _apply_snn_train_flags(state, flags)

def load_modulated_checkpoint(ckpt_path: Union[str, Path], settings: Dict, quiet: bool = False) -> Tuple[Dict[str, torch.Tensor], nn.Module]:
    ckpt = torch.load(ckpt_path, map_location=device)
    snn_params = ckpt.get("snn_params")
    mlp_state = ckpt.get("mlp_state")
    if snn_params is None or mlp_state is None:
        raise ValueError(f"Checkpoint at {ckpt_path} is missing SNN or MLP parameters.")
    state = {k: torch.nn.Parameter(v.to(device), requires_grad=False) for k, v in snn_params.items()}
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

# =============================================================================
# SECTION 4: Modulated Eval, Spike Stats, and SHAP
# =============================================================================
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
        print("SHAP skipped: snn_add mode does not expose MLP blocks.")
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

# =============================================================================
# SECTION 5: Modulated Training
# =============================================================================

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

# =============================================================================
# SECTION 6: Command Line Interface
# =============================================================================

def _parse_mask(disable_csv: str, all_keys: list) -> dict:
    disabled = set([k.strip() for k in disable_csv.split(",") if k.strip()]) if disable_csv else set()
    return {k: (k not in disabled) for k in all_keys}

def main(argv: Optional[Sequence[str]] = None):
    p = argparse.ArgumentParser(description="Unified SNN / Modulated SNN runner")

    # --- core model/dataset/sim ---
    p.add_argument("--run_mode", type=str, default="mod", choices=["snn", "mod", "staged"])
    p.add_argument("--nb_inputs", type=int, default=700)
    p.add_argument("--nb_hidden", type=int, default=256)
    p.add_argument("--nb_outputs", type=int, default=20)
    p.add_argument("--time_step", type=float, default=1e-3)
    p.add_argument("--nb_steps", type=int, default=100)
    p.add_argument("--max_time", type=float, default=1.4)
    p.add_argument("--batch_size", type=int, default=64)

    p.add_argument("--cache_dir", type=str, default="~/data")
    p.add_argument("--cache_subdir", type=str, default="hdspikes")
    p.add_argument("--train_file", type=str, default="shd_train.h5")
    p.add_argument("--test_file", type=str, default="shd_test.h5")
    p.add_argument("--val_file", type=str, default=None,
                   help="Optional validation H5 file; if provided, skips splitting train set.")

    # --- training schedule/hparams ---
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--tau_syn", type=float, default=10e-3)
    p.add_argument("--tau_mem", type=float, default=20e-3)
    p.add_argument("--tau_match_clip", type=str2bool, nargs="?", const=True, default=False,
                   help="If True, convert tau_syn/tau_mem using max_time/nb_steps so they follow clip duration.")
    p.add_argument("--weight_scale", type=float, default=0.2)
    p.add_argument("--psp_norm_peak", type=str2bool, nargs="?", const=True, default=False,
                   help="Normalize bi-exponential PSP peaks to 1 (scales syn->mem and output filters).")

    # single-phase default epochs (used for run_mode snn/mod, and as fallback)
    p.add_argument("--nb_epochs", type=int, default=30)

    # staged-specific epoch counts
    p.add_argument("--nb_epochs_snn", type=int, default=30, help="Epochs for Stage-1 SNN in staged mode.")
    p.add_argument("--nb_epochs_mod", type=int, default=70, help="Epochs for Stage-2 Mod in staged mode.")


    # --- save dirs ---
    p.add_argument("--save_dir_root", type=str, default=None,
                   help="Shared base directory; SNN and Mod runs will be stored under SNN/ and NM/ subfolders.")
    p.add_argument("--save_dir_snn", type=str, default="Runs_SNN_EXP")
    p.add_argument("--save_dir_mod", type=str, default="Runs_Mod_EXP")

    # --- loss regularization toggles ---
    p.add_argument("--snn_reg_enable", type=str2bool, nargs="?", const=True, default=True,
                   help="Enable spike regularization term during SNN training. Default=True.")
    p.add_argument("--mod_reg_enable", type=str2bool, nargs="?", const=True, default=True,
                   help="Enable spike regularization term during modulated training. Default=True.")
    p.add_argument("--snn_mod_reg_enable", type=str2bool, nargs="?", const=True, default=False,
                   help="Enable spike regularization term for the modulator SNN (snn_add). Default=False.")
    p.add_argument("--snn_reg_scale", type=float, default=1.0,
                   help="Scalar multiplier applied to the SNN regularization term. Default=1.0.")
    p.add_argument("--mod_reg_scale", type=float, default=1.0,
                   help="Scalar multiplier applied to the Mod regularization term. Default=1.0.")
    p.add_argument("--snn_mod_reg_scale", type=float, default=1.0,
                   help="Scalar multiplier applied to the modulator-SNN regularization term. Default=1.0.")

    # --- augmentation (train) ---
    p.add_argument("--aug_jitter_ms", type=float, default=3.0)
    p.add_argument("--aug_shift_ms", type=float, default=15.0)
    p.add_argument("--aug_scale_low", type=float, default=0.95)
    p.add_argument("--aug_scale_high", type=float, default=1.05)
    p.add_argument("--aug_drop_p", type=float, default=0.10)
    p.add_argument("--aug_insert_rate", type=float, default=0.005)
    p.add_argument("--aug_band_frac", type=float, default=0.05)
    p.add_argument("--aug_channel_jitter_std", type=float, default=20.0,
                   help="Std dev (channels) for Gaussian jitter of input unit index (sigma_u in paper).")
    p.add_argument("--aug_noise_rate_hz", type=float, default=0.0,
                   help="Poisson background noise rate (spikes/s) injected per sample, as in Heidelberg paper.")
    p.add_argument("--aug_noise_per_input", type=str2bool, nargs="?", const=True, default=False,
                   help="Interpret aug_noise_rate_hz as a per-input rate (default: global rate).")
    p.add_argument("--channel_compress_enable", type=str2bool, nargs="?", const=True, default=False,
                   help="Enable channel compression (coarse-grain neighboring input channels).")
    p.add_argument("--channel_compress_target", type=int, default=70,
                   help="Target number of input channels when compression is enabled (paper used 70 from 700 via groups of 10).")
    p.add_argument("--channel_compress_mode", type=str, default="all",
                   choices=["all", "mod_only", "mod_mlp"],
                   help="Where to apply compression: all inputs (default), mod_only (modulator only), mod_mlp (learned MLP compression for modulator).")
    p.add_argument("--channel_compress_mlp_hidden_sizes", type=str, default="[]",
                   help="Hidden sizes for modulator input compression MLP when channel_compress_mode=mod_mlp (e.g., \"[256]\").")
    p.add_argument("--postbin_time_mask_train", type=float, default=0.05)
    p.add_argument("--hidden_dropout_p", type=float, default=0.0,
                   help="Dropout prob on hidden spikes during training (0 disables).")

    # --- augmentation toggles ---
    p.add_argument("--train_aug_enable", type=str2bool, nargs="?", const=True, default=False,
                   help="Enable augmentation during TRAIN. Default=False.")
    p.add_argument("--eval_aug_enable",  type=str2bool, nargs="?", const=True, default=False,
                   help="Enable augmentation during VALIDATION. Default=False.")
    p.add_argument("--test_aug_enable",  type=str2bool, nargs="?", const=True, default=False,
                   help="Enable augmentation during TEST. Default=False.")
    p.add_argument("--paper_aug_train", type=str2bool, nargs="?", const=True, default=False,
                   help="Use only paper-style aug/noise (channel jitter + optional Poisson) for TRAIN; disables legacy aug knobs.")
    p.add_argument("--paper_aug_eval", type=str2bool, nargs="?", const=True, default=False,
                   help="Use only paper-style aug/noise for VALIDATION; disables legacy aug knobs.")
    p.add_argument("--paper_aug_test", type=str2bool, nargs="?", const=True, default=False,
                   help="Use only paper-style aug/noise for TEST; disables legacy aug knobs.")
    p.add_argument("--paper_tau_scale_enable", type=str2bool, nargs="?", const=True, default=False,
                   help="Enable time-constant scaling factor (paper best: 4x).")
    p.add_argument("--paper_tau_scale", type=float, default=4.0,
                   help="Scaling factor applied to tau_mem and tau_syn when enabled.")
    p.add_argument("--train_noise_enable", type=str2bool, nargs="?", const=True, default=False,
                   help="Enable Poisson noise injection during TRAIN split.")
    p.add_argument("--eval_noise_enable", type=str2bool, nargs="?", const=True, default=False,
                   help="Enable Poisson noise injection during VALIDATION split.")
    p.add_argument("--test_noise_enable", type=str2bool, nargs="?", const=True, default=False,
                   help="Enable Poisson noise injection during TEST split.")

    # --- test-time postbin mask (independent from eval) ---
    p.add_argument("--postbin_time_mask_test", type=float, default=0.0)


    # --- eval augmentation (default: clean/no aug) ---
    # p.add_argument("--eval_aug_enable", action="store_true", help="If set, use same aug knobs for eval; default off")
    p.add_argument("--postbin_time_mask_eval", type=float, default=0.0)

    # --- split / training controls ---
    p.add_argument("--use_validation", type=str2bool, nargs="?", const=True, default=True,
                    help="Whether to use validation (True/False). Default=True.")
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--k_folds", type=int, default=1)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--fixed_split_path", type=str, default=None)
    p.add_argument("--reinit_per_fold", type=str2bool, nargs="?", const=True, default=True,
                    help="Reinitialize model per fold (True/False). Default=True.")

    # --- staged controls ---
    p.add_argument("--staged_unlock_metric", type=str, default="train_acc", choices=["train_acc", "val_acc"])
    p.add_argument("--staged_unlock_threshold", type=float, default=0.85)
    p.add_argument("--staged_min_epochs", type=int, default=1)

    # --- mod-only / MLP ---
    p.add_argument("--start_locked", type=str2bool, nargs="?", const=True, default=False,
                    help="Lock SNN params at start of mod training (True/False). Default=False.")
    p.add_argument("--unlock_metric", type=str, default=None, choices=[None, "val_acc", "train_acc"])
    p.add_argument("--unlock_threshold", type=float, default=None)

    # --- Modulator (ANN) configuration ---
    p.add_argument("--ann_hidden_sizes", type=str, default="[2048]",
                   help="List of hidden sizes for the ANN/RNN/LSTM modulator (e.g. \"[2048, 1024]\").")
    p.add_argument("--ann_interval", type=int, default=3,
                   help="Cadence (in steps) between modulation updates.")
    p.add_argument("--ann_mode", type=str, default="ann_sub",
                   choices=["ann_sub", "ann_add", "snn_add"],
                   help="Modulation strategy: ann_sub (~mlp_sub), ann_add (~mlp_add), or snn_add.")
    p.add_argument("--ann_arch", type=str, default="mlp", choices=["mlp"],
                   help="Backbone used for ann_sub/ann_add modulators.")
    p.add_argument("--mod_update_every_step", type=str2bool, nargs="?", const=True, default=False,
                   help="If True, apply modulation updates every timestep instead of only on ann_interval.")
    p.add_argument("--mod_current_enable", type=str2bool, nargs="?", const=True, default=False,
                   help="Use modulator outputs as injected currents instead of parameter updates.")
    p.add_argument("--mod_current_target", type=str, default="both",
                   choices=["hidden", "output", "both"],
                   help="Where to inject modulator currents when mod_current_enable is true.")
    p.add_argument("--mod_current_activation", type=str, default="tanh",
                   choices=["sigmoid", "tanh"],
                   help="Activation on modulator current outputs when mod_current_enable is true.")
    p.add_argument("--mod_hidden_sizes", type=str, default=None,
                   help="Hidden layer sizes for the modulator (falls back to --ann_hidden_sizes when omitted).")
    p.add_argument("--snn_mod_hidden", type=int, default=None,
                   help="(Deprecated) Number of modulator neurons. In snn_add mode this is derived automatically (10*nb_hidden + 4*nb_outputs).")
    p.add_argument("--snn_mod_hidden_recurrent", type=str2bool, nargs="?", const=True, default=False,
                   help="Enable recurrent connections inside secondary SNN hidden layers.")
    p.add_argument("--snn_mod_rec_init_zero", type=str2bool, nargs="?", const=True, default=True,
                   help="If True and snn_mod_hidden_recurrent is enabled, initialize modulator SNN recurrent weights to 0. Default=True.")
    p.add_argument("--snn_mod_gain_init", type=float, default=SNN_ADD_GAIN_INIT_DEFAULT,
                   help="Initial gain per modulator neuron for snn_add mode (small near-zero start).")
    p.add_argument("--snn_mod_weight_scale", type=float, default=SNN_ADD_WEIGHT_SCALE_DEFAULT,
                   help="Scale used when initializing snn_add input weights (small near-zero default).")
    p.add_argument("--snn_add_balanced_init", type=str2bool, nargs="?", const=True, default=False,
                   help="If True (snn_add), initialize paired +/- modulator neurons to cancel initially (pass-through start). Default=False.")
    p.add_argument("--snn_add_init_effect_frac", type=float, default=0.05,
                   help="When snn_add_balanced_init is True, set per-side gain to this fraction of each parameter's range width (net still cancels). Default=0.05.")
    # Neuromodulator-inspired modulation knobs.
    p.add_argument("--nm_enable", type=str2bool, nargs="?", const=True, default=False,
                   help="Enable neuromodulator-based modulation (snn_add/ann_sub/ann_add).")
    p.add_argument("--nm_counts", type=str, default="[0,0]",
                   help="List of neuromodulator type counts per layer (e.g. \"[6,5]\" for hidden/output).")
    p.add_argument("--nm_init_scale", type=float, default=SOFT_MOD_GAIN_INIT,
                   help="Scale for initializing shared neuromodulator MLP weights (small near-zero default).")
    p.add_argument("--nm_neuron_frac_enable", type=str2bool, nargs="?", const=True, default=False,
                   help="Enable random neuron masking for NM effects (fractions per layer).")
    p.add_argument("--nm_neuron_frac", type=str, default="[1.0,1.0]",
                   help="Fractions of neurons modulated in [hidden, output] (e.g., \"[0.8,0.6]\").")
    p.add_argument("--nm_param_frac_enable", type=str2bool, nargs="?", const=True, default=False,
                   help="Enable random parameter masking per neuron for NM effects.")
    p.add_argument(
        "--nm_param_frac",
        type=str,
        default='{"alpha_1":1.0,"beta_1":1.0,"thr":1.0,"reset":1.0,"rest":1.0,"alpha_2":1.0,"beta_2":1.0}',
                   help="Parameter fractions per neuron. Accepts [hidden, output] or a dict like "
                        "'{\"alpha_1\":0.5,\"beta_1\":0.5,\"thr\":0.5,\"reset\":0.5,"
                        "\"rest\":0.5,\"alpha_2\":1.0,\"beta_2\":1.0}'.")
    p.add_argument("--nm_mapper_type", type=str, default="mlp",
                   choices=["mlp", "linear"],
                   help="Neuromodulator mapper type: full MLP (default) or single linear layer.")
    p.add_argument("--nm_mapper_activation", type=str, default="auto",
                   choices=["auto", "linear", "sigmoid", "tanh", "none"],
                   help="Final activation for NM mapper. 'auto' matches mode (sigmoid for sub, tanh for add).")
    p.add_argument("--nm_mapper_hidden_activation", type=str, default="silu",
                   choices=["silu", "gelu", "leakyrelu", "tanh", "relu", "none"],
                   help="Hidden activation for NM mapper MLP (ignored for linear mapper).")
    p.add_argument("--nm_mapper_hidden_size", type=int, default=None,
                   help="Override NM mapper hidden width (applies to both hidden/output mapper MLPs).")
    p.add_argument("--nm_flat_order", type=str, default="type_major",
                   choices=["type_major", "target_major"],
                   help="How NM levels are packed in the flat modulator output. "
                        "type_major packs by modulator type then target index; target_major packs by target then type.")
    p.add_argument("--nm_warm_start", type=str2bool, nargs="?", const=True, default=True,
                   help="Warm-start NM to behave like normal mode (best with --nm_counts [5,2]). "
                        "Enables identity init for the NM mapper and sets flat order to type_major unless overridden.")
    p.add_argument("--nm_warm_start_force_linear", type=str2bool, nargs="?", const=True, default=False,
                   help="When nm_warm_start is true and nm_counts is [5,2], force nm_mapper_type=linear for the closest "
                        "match to normal mode. Set false to keep an MLP mapper (optionally with nm_mapper_hidden_activation=none).")
    p.add_argument("--mod_fixed_mask_enable", type=str2bool, nargs="?", const=True, default=False,
                   help="Use fixed neuron/param masks for the whole run and shrink modulator IO (applies to NM and non-NM).")
    p.add_argument("--mod_fixed_mask_seed", type=int, default=None,
                   help="Seed for fixed modulation mask sampling (defaults to --seed when unset).")
    p.add_argument("--mod_fixed_mask_flat_inputs", type=str2bool, nargs="?", const=True, default=False,
                   help="Also reduce hid_flat/out_flat inputs using the fixed neuron mask.")
    p.add_argument("--mod_fixed_mask_zero_fallback", type=str2bool, nargs="?", const=True, default=False,
                   help="Legacy behavior: if True, 0.0 fractions fall back to full selection instead of empty.")
    p.add_argument("--mod_hid_flat_group", type=str2bool, nargs="?", const=True, default=False,
                   help="Group-compress hid_flat inputs using the hidden group layout.")
    p.add_argument("--mod_hid_flat_modulated_only", type=str2bool, nargs="?", const=True, default=False,
                   help="Limit hid_flat inputs to neurons/groups with any modulated parameter (requires fixed mask).")

    # Block masking helpers (comma-separated lists of ANN inputs/outputs to disable).
    p.add_argument("--ann_in_disable", type=str, default="",
                   help="Comma-separated ANN input blocks to disable.")
    p.add_argument("--ann_out_disable", type=str, default="",
                   help="Comma-separated ANN output blocks to disable.")
    p.add_argument("--ann_output_activation", type=str.lower, default=None,
                   choices=list(ANN_OUTPUT_ACTIVATIONS.keys()) + ["none", "default"],
                   help="Optional ANN/RNN/LSTM output activation override (linear/sigmoid/tanh); use 'default' or 'none' to keep the mode-specific default.")

    # Grouping controls: share modulation deltas across overlapping neurons (format [hidden, output]).
    p.add_argument("--group_size", type=str, nargs="+", default=["1", "1"],
                   help="Grouping window size for [hidden, output] layers.")
    p.add_argument("--group_overlap", type=str, nargs="+", default=["0", "0"],
                   help="Number of neurons of overlap for [hidden, output] groups (uniform: overlapping contributions are averaged).")
    p.add_argument("--group_distribution", type=str, nargs="+", default=["uniform", "uniform"],
                   help="Distribution per layer (\"uniform\" or \"normal\"). Normal uses soft Gaussian falloff; overlap is ignored.")
    p.add_argument("--group_normal_std", type=str, nargs="+", default=["1.0", "1.0"],
                   help="Stddev when distribution is 'normal' for [hidden, output] (space-separated or JSON list).")

    # Per-parameter modulation cadence (learned offsets).
    p.add_argument("--param_timescales_enable", type=str2bool, nargs="?", const=True, default=False,
                   help="Enable learnable per-parameter modulation intervals.")
    p.add_argument("--param_timescales_distribution", type=str, default="fixed",
                   choices=["fixed", "uniform", "normal"])
    p.add_argument("--param_timescales_scale", type=float, default=0.0,
                   help="Scale for initializing learnable intervals when distribution != fixed.")
    p.add_argument("--param_timescales_std", type=float, default=0.0,
                   help="Stddev for timescale offsets (used when distribution=normal; falls back to scale).")
    p.add_argument("--param_timescales_seed", type=int, default=None,
                   help="Optional RNG seed for initializing learnable intervals.")
    p.add_argument("--param_timescales_trainable", type=str2bool, nargs="?", const=True, default=True,
                   help="Toggle requires_grad for param timescale offsets.")
    # Per-parameter modulation smoothing (learned time constants).
    p.add_argument("--param_smoothing_enable", type=str2bool, nargs="?", const=True, default=False,
                   help="Enable learned time-constant smoothing of modulation updates.")
    p.add_argument("--param_smoothing_tau_init", type=float, default=PARAM_SMOOTH_TAU_INIT_DEFAULT,
                   help="Initial per-step mixing fraction (0-1) for smoothing modulation outputs (e.g., 0.2 -> 20% toward target per step).")
    p.add_argument("--param_smoothing_tau_min", type=float, default=PARAM_SMOOTH_TAU_MIN_DEFAULT,
                   help="Minimum per-step mixing fraction (keeps smoothing from becoming zero).")
    p.add_argument("--param_smoothing_trainable", type=str2bool, nargs="?", const=True, default=True,
                   help="Toggle requires_grad for smoothing time constants.")
    p.add_argument("--param_smoothing_modes", type=str, nargs="+",
                   default=None,
                   choices=["ann_sub", "ann_add", "snn_add", "mlp_sub", "mlp_add", "all"],
                   help="Optional subset of modes to smooth; omit to smooth all when enabled.")

    # Parameter range overrides (used for clamps and initialisation).
    p.add_argument("--alpha_1_range", type=float, nargs=2, metavar=("MIN", "MAX"), default=None,
                   help="Allowed range for alpha_1 (default ~[1/e,0.995]).")
    p.add_argument("--beta_1_range", type=float, nargs=2, metavar=("MIN", "MAX"), default=None,
                   help="Allowed range for beta_1 (default ~[1/e,0.995]).")
    p.add_argument("--thr_range", type=float, nargs=2, metavar=("MIN", "MAX"), default=None,
                   help="Allowed range for thresholds (default [0.5,1.5]).")
    p.add_argument("--reset_range", type=float, nargs=2, metavar=("MIN", "MAX"), default=None,
                   help="Allowed range for reset potentials (default [-0.5,0.5]).")
    p.add_argument("--rest_range", type=float, nargs=2, metavar=("MIN", "MAX"), default=None,
                   help="Allowed range for rest potentials (default [-0.5,0.5]).")
    p.add_argument("--alpha_2_range", type=float, nargs=2, metavar=("MIN", "MAX"), default=None,
                   help="Allowed range for alpha_2 (default ~[1/e,0.995]).")
    p.add_argument("--beta_2_range", type=float, nargs=2, metavar=("MIN", "MAX"), default=None,
                   help="Allowed range for beta_2 (default ~[1/e,0.995]).")

    # --- SHAP analysis toggles ---
    p.add_argument("--ann_shap_enable", type=str2bool, nargs="?", const=True, default=False,
                   help="Approximate Shapley importance for ANN input/output blocks after mod/staged runs.")
    p.add_argument("--ann_shap_samples", type=int, default=16,
                   help="Number of random permutations for SHAP approximation (per input/output group).")
    p.add_argument("--ann_shap_batch_limit", type=int, default=4,
                   help="How many evaluation batches to cache for SHAP metric sweeps.")
    p.add_argument("--ann_shap_dataset", type=str, default="test", choices=["train", "test"],
                   help="Dataset split to score during SHAP analysis.")
    p.add_argument("--ann_shap_metric", type=str, default="acc", choices=["acc", "nll"],
                   help="Metric to explain (accuracy or NLL).")

    # --- base ckpt for --run_mode=mod ---
    p.add_argument("--base_snn_ckpt", type=str, default=None, help="Path to base SNN checkpoint for 'mod' mode")
    p.add_argument("--base_snn_from_stockpile", type=str2bool, nargs="?", const=True, default=False,
                   help="If True (mod mode), ignore --base_snn_ckpt and sample a base SNN from a stockpile directory.")
    p.add_argument("--base_snn_stockpile_dir", type=str, default=None,
                   help="Directory containing Run_* folders with base SNN checkpoints (used with --base_snn_from_stockpile).")
    p.add_argument("--base_snn_stockpile_seed", type=int, default=None,
                   help="Optional seed to make stockpile checkpoint selection reproducible (default: None = random).")

    p.add_argument("--save_every_epoch", type=str2bool, nargs="?", const=True, default=True,
                    help="Save checkpoints every epoch (True/False). Default=True.")

    p.add_argument("--homo_init", type=str2bool, nargs="?", const=True, default=False,
                   help="If True, initialize neuron params homogeneously (thr=1, reset=rest=0, taus from base).")

    # --- SNN parameter training toggles ---
    p.add_argument("--snn_train_disable", type=str, default="",
                   help="Comma-separated list of SNN params to freeze (options: w1,w2,v1,alpha_hetero_1,beta_hetero_1,"
                        "alpha_hetero_2,beta_hetero_2,thresholds_1,reset_1,rest_1).")

    args = p.parse_args(argv)

    # Normalize save directories
    if args.save_dir_root is not None:
        root_str = str(args.save_dir_root).strip()
        if root_str and root_str.lower() != "none":
            root_path = Path(os.path.expanduser(root_str))
            args.save_dir_root = str(root_path)
            args.save_dir_snn = str(root_path / "SNN")
            args.save_dir_mod = str(root_path / "NM")
        else:
            args.save_dir_root = None

    shap_cfg = dict(
        enable=args.ann_shap_enable,
        samples=args.ann_shap_samples,
        batch_limit=args.ann_shap_batch_limit,
        dataset=args.ann_shap_dataset,
        metric=args.ann_shap_metric
    )

    # --- settings dict assembly (kept same names) ---
    # effective taus (paper scaling)
    tau_scale = args.paper_tau_scale if args.paper_tau_scale_enable else 1.0
    tau_syn_eff = args.tau_syn * tau_scale
    tau_mem_eff = args.tau_mem * tau_scale

    # channel compression
    nb_inputs_raw = args.nb_inputs
    compress_factor = 1
    nb_inputs_comp = nb_inputs_raw
    if args.channel_compress_enable:
        target = max(1, min(int(args.channel_compress_target), nb_inputs_raw))
        compress_factor = max(1, int(math.ceil(nb_inputs_raw / target)))
        nb_inputs_comp = int(math.ceil(nb_inputs_raw / compress_factor))
    compress_mode = str(args.channel_compress_mode).lower()
    if not args.channel_compress_enable:
        compress_mode = "none"
    nb_inputs_snn = nb_inputs_raw
    nb_inputs_mod = nb_inputs_raw
    compress_factor_snn = 1
    compress_factor_mod = 1
    if compress_mode == "all":
        nb_inputs_snn = nb_inputs_comp
        nb_inputs_mod = nb_inputs_comp
        compress_factor_snn = compress_factor
        compress_factor_mod = compress_factor
    elif compress_mode == "mod_only":
        nb_inputs_snn = nb_inputs_raw
        nb_inputs_mod = nb_inputs_comp
        compress_factor_snn = 1
        compress_factor_mod = compress_factor
    elif compress_mode == "mod_mlp":
        nb_inputs_snn = nb_inputs_raw
        nb_inputs_mod = nb_inputs_comp
        compress_factor_snn = 1
        compress_factor_mod = 1

    compress_mlp_hidden_sizes = parse_int_list(args.channel_compress_mlp_hidden_sizes)
    if compress_mlp_hidden_sizes is None:
        compress_mlp_hidden_sizes = []

    settings = dict(
        nb_inputs=nb_inputs_snn,
        nb_inputs_raw=nb_inputs_raw,
        nb_inputs_mod=nb_inputs_mod,
        channel_compress_enable=bool(args.channel_compress_enable),
        channel_compress_mode=compress_mode,
        channel_compress_target=int(args.channel_compress_target),
        channel_compress_factor=int(compress_factor),
        channel_compress_factor_snn=int(compress_factor_snn),
        channel_compress_factor_mod=int(compress_factor_mod),
        channel_compress_mlp_hidden_sizes=compress_mlp_hidden_sizes,

        nb_hidden=args.nb_hidden,
        nb_outputs=args.nb_outputs,

        time_step=args.time_step,
        nb_steps=args.nb_steps,
        max_time=args.max_time,

        batch_size=args.batch_size,
        cache_dir=args.cache_dir,
        cache_subdir=args.cache_subdir,
        train_file=args.train_file,
        test_file=args.test_file,

        lr=args.lr,
        nb_epochs=args.nb_epochs,

        tau_syn=tau_syn_eff,
        tau_mem=tau_mem_eff,
        tau_match_clip=args.tau_match_clip,

        psp_norm_peak=args.psp_norm_peak,
        weight_scale=args.weight_scale,

        save_dir_snn=args.save_dir_snn,
        save_dir_mod=args.save_dir_mod,
        save_dir_root=args.save_dir_root,
        paper_tau_scale=tau_scale,
        paper_tau_scale_enable=bool(args.paper_tau_scale_enable),
        homo_init=bool(args.homo_init),
        hidden_dropout_p=float(args.hidden_dropout_p),
    )

    disabled_params = set()
    if args.snn_train_disable:
        disabled_params = {p.strip() for p in args.snn_train_disable.split(",") if p.strip()}
    invalid = sorted(disabled_params - set(SNN_PARAM_NAMES))
    if invalid:
        raise ValueError(f"Unknown SNN parameters in --snn_train_disable: {', '.join(invalid)}")
    settings["snn_train_flags"] = {name: (name not in disabled_params) for name in SNN_PARAM_NAMES}

    # --- settings dict assembly for augmentation ---
    def _build_aug_cfg(enable_aug: bool, enable_noise: bool, paper_only: bool = False):
        compress_factor = settings.get("channel_compress_factor_snn", settings.get("channel_compress_factor", 1))
        if paper_only:
            if not (enable_aug or enable_noise or compress_factor > 1):
                return None
            eff_factor = max(1, compress_factor)
            channel_jitter_std = args.aug_channel_jitter_std if (enable_aug or enable_noise) else 0.0
            noise_rate_hz = args.aug_noise_rate_hz if enable_noise else 0.0
            return dict(
                jitter_ms=0.0, shift_ms=0.0, scale_low=1.0, scale_high=1.0,
                drop_p=0.0, insert_rate=0.0, band_frac=0.0,
                compress_factor=compress_factor,
                channel_jitter_std=channel_jitter_std,
                channel_jitter_std_eff=channel_jitter_std / float(eff_factor),
                noise_rate_hz=noise_rate_hz,
                noise_rate_hz_eff=noise_rate_hz * (eff_factor if args.aug_noise_per_input and enable_noise else 1.0),
                noise_per_input=args.aug_noise_per_input if enable_noise else False,
            )
        if not (enable_aug or enable_noise or compress_factor > 1):
            return None
        eff_factor = max(1, compress_factor)
        channel_jitter_std = args.aug_channel_jitter_std if enable_noise else 0.0
        noise_rate_hz = args.aug_noise_rate_hz if enable_noise else 0.0
        return dict(
            jitter_ms=args.aug_jitter_ms if enable_aug else 0.0,
            shift_ms=args.aug_shift_ms if enable_aug else 0.0,
            scale_low=args.aug_scale_low if enable_aug else 1.0,
            scale_high=args.aug_scale_high if enable_aug else 1.0,
            drop_p=args.aug_drop_p if enable_aug else 0.0,
            insert_rate=args.aug_insert_rate if enable_aug else 0.0,
            band_frac=args.aug_band_frac if enable_aug else 0.0,
            compress_factor=compress_factor,
            channel_jitter_std=channel_jitter_std,
            channel_jitter_std_eff=channel_jitter_std / float(eff_factor),
            noise_rate_hz=noise_rate_hz,
            noise_rate_hz_eff=noise_rate_hz * (eff_factor if args.aug_noise_per_input and enable_noise else 1.0),
            noise_per_input=args.aug_noise_per_input if enable_noise else False,
        )

    AUGMENT_TRAIN = _build_aug_cfg(args.train_aug_enable, args.train_noise_enable, paper_only=args.paper_aug_train)
    POSTBIN_TIME_MASK_TRAIN = args.postbin_time_mask_train if (args.train_aug_enable or args.train_noise_enable) else 0.0

    AUGMENT_EVAL = _build_aug_cfg(args.eval_aug_enable, args.eval_noise_enable, paper_only=args.paper_aug_eval)
    POSTBIN_TIME_MASK_EVAL = args.postbin_time_mask_eval

    AUGMENT_TEST = _build_aug_cfg(args.test_aug_enable, args.test_noise_enable, paper_only=args.paper_aug_test)
    POSTBIN_TIME_MASK_TEST = args.postbin_time_mask_test if (args.test_aug_enable or args.test_noise_enable) else 0.0

    settings.update(dict(
        augment_train=AUGMENT_TRAIN,
        postbin_mask_train=POSTBIN_TIME_MASK_TRAIN,
        augment_eval=AUGMENT_EVAL,
        postbin_mask_eval=POSTBIN_TIME_MASK_EVAL,
        augment_test=AUGMENT_TEST,
        postbin_mask_test=POSTBIN_TIME_MASK_TEST
    ))

    settings.update({
        "use_snn_reg": args.snn_reg_enable,
        "use_mod_reg": args.mod_reg_enable,
        "use_snn_mod_reg": args.snn_mod_reg_enable,
        "snn_reg_scale": args.snn_reg_scale,
        "mod_reg_scale": args.mod_reg_scale,
        "snn_mod_reg_scale": args.snn_mod_reg_scale,
    })


    # run mode
    run_mode = args.run_mode

    # MLP masks
    in_all  = ["alpha_1","beta_1","thr","reset","rest","alpha_2","beta_2","in_flat","hid_flat","out_flat"]
    out_all = ["alpha_1","beta_1","thr","reset","rest","alpha_2","beta_2"]
    mlp_in_mask  = _parse_mask(args.ann_in_disable, in_all)
    mlp_out_mask = _parse_mask(args.ann_out_disable, out_all)
    mlp_hidden_sizes = parse_int_list(args.ann_hidden_sizes)
    if mlp_hidden_sizes is None:
        mlp_hidden_sizes = [2048]
    mod_hidden_sizes = parse_int_list(args.mod_hidden_sizes)
    if mod_hidden_sizes is None:
        mod_hidden_sizes = mlp_hidden_sizes
    mlp_interval = args.ann_interval
    mlp_mode = args.ann_mode
    mlp_arch = args.ann_arch
    ann_activation_raw = args.ann_output_activation
    ann_activation = None
    if isinstance(ann_activation_raw, str):
        ann_activation_raw = ann_activation_raw.strip()
        lowered = ann_activation_raw.lower()
        if ann_activation_raw and lowered not in {"none", "default"}:
            ann_activation = lowered

    normalized_mlp_mode = _normalize_mlp_mode(mlp_mode)
    resolved_activation = _final_activation_for_mode(normalized_mlp_mode, ann_activation)[1]

    # MLP config
    settings.update({
        "mlp_hidden_sizes": mlp_hidden_sizes,
        "mod_hidden_sizes": mod_hidden_sizes,
        "mlp_in_mask": mlp_in_mask,
        "mlp_out_mask": mlp_out_mask,
        "mlp_interval": mlp_interval,
        "ann_interval": mlp_interval,
        "mlp_mode": mlp_mode,
        "ann_mode": args.ann_mode,
        "mlp_arch": mlp_arch,
        "snn_mod_hidden": args.snn_mod_hidden,
        "snn_mod_gain_init": args.snn_mod_gain_init,
        "snn_mod_weight_scale": args.snn_mod_weight_scale,
        "snn_mod_hidden_recurrent": args.snn_mod_hidden_recurrent,
        "snn_mod_rec_init_zero": bool(args.snn_mod_rec_init_zero),
        "snn_add_balanced_init": bool(args.snn_add_balanced_init),
        "snn_add_init_effect_frac": float(args.snn_add_init_effect_frac),
        "ann_output_activation": ann_activation,
        "ann_output_activation_effective": resolved_activation,
        "mod_update_every_step": bool(args.mod_update_every_step),
        "mod_current_enable": bool(args.mod_current_enable),
        "mod_current_target": str(args.mod_current_target),
        "mod_current_activation": str(args.mod_current_activation),
    })
    settings["mlp_mode"] = normalized_mlp_mode

    # Parameter ranges
    param_ranges = {}
    for name in ("alpha_1", "beta_1", "thr", "reset", "rest", "alpha_2", "beta_2"):
        values = getattr(args, f"{name}_range")
        if values:
            param_ranges[name] = (float(values[0]), float(values[1]))
    if param_ranges:
        settings["param_ranges"] = param_ranges

    group_size = _parse_int_pair(args.group_size) or (1, 1)
    group_overlap = _parse_int_pair(args.group_overlap) or (0, 0)
    group_distribution = _parse_str_pair(args.group_distribution) or ("uniform", "uniform")
    group_normal_std = _parse_float_pair(args.group_normal_std) or (1.0, 1.0)
    hidden_size, output_size = group_size
    hidden_overlap, output_overlap = group_overlap
    group_enable = (hidden_size > 1 or hidden_overlap > 0,
                    output_size > 1 or output_overlap > 0)
    settings["group_cfg"] = {
        "enable": group_enable,
        "size": group_size,
        "overlap": group_overlap,
        "distribution": tuple(d.lower() for d in group_distribution),
        "normal_std": group_normal_std,
    }

    nm_counts = parse_int_list(args.nm_counts)
    if not nm_counts:
        nm_counts = [0, 0]
    hidden_nm = int(nm_counts[0]) if len(nm_counts) > 0 else 0
    output_nm = int(nm_counts[1]) if len(nm_counts) > 1 else 0
    nm_activation_kind = None
    nm_mapper_activation = str(getattr(args, "nm_mapper_activation", "auto") or "auto").lower()
    if nm_mapper_activation == "auto":
        if normalized_mlp_mode in {"mlp_add", "snn_add"}:
            nm_activation_kind = "tanh"
        elif normalized_mlp_mode in {"mlp_sub"}:
            nm_activation_kind = "sigmoid"
    else:
        if nm_mapper_activation not in {"none", "linear"}:
            nm_activation_kind = nm_mapper_activation
    nm_neuron_frac = parse_float_list(args.nm_neuron_frac) or [1.0, 1.0]
    if len(nm_neuron_frac) == 1:
        nm_neuron_frac = [nm_neuron_frac[0], nm_neuron_frac[0]]
    nm_param_frac_map = _parse_param_fraction_map(args.nm_param_frac, default=1.0)
    nm_cfg: Dict[str, object] = {
        "enable": args.nm_enable,
        "hidden_per_neuron": max(0, hidden_nm),
        "output_per_neuron": max(0, output_nm),
        "init_scale": args.nm_init_scale,
        "activation_kind": nm_activation_kind,
        "mapper_type": str(getattr(args, "nm_mapper_type", "mlp") or "mlp").lower(),
        "hidden_activation": str(getattr(args, "nm_mapper_hidden_activation", "silu") or "silu").lower(),
        "flat_order": str(getattr(args, "nm_flat_order", "type_major") or "type_major").lower(),
        "warm_start": bool(getattr(args, "nm_warm_start", False)),
        "warm_start_force_linear": bool(getattr(args, "nm_warm_start_force_linear", False)),
        "neuron_fraction_enable": args.nm_neuron_frac_enable,
        "neuron_fraction": (float(nm_neuron_frac[0]), float(nm_neuron_frac[1])) if len(nm_neuron_frac) >= 2 else (1.0, 1.0),
        "param_fraction_enable": args.nm_param_frac_enable,
        "param_fraction": nm_param_frac_map,
    }
    if nm_cfg.get("warm_start"):
        nm_cfg["init_identity"] = True
        # If counts match the parameter set sizes, a linear identity mapper is the closest analogue to normal mode.
        if bool(nm_cfg.get("warm_start_force_linear", True)) and hidden_nm == 5 and output_nm == 2:
            nm_cfg["mapper_type"] = "linear"
    nm_mapper_hidden_size = getattr(args, "nm_mapper_hidden_size", None)
    if nm_mapper_hidden_size is not None:
        nm_cfg["hidden_mlp_hidden"] = int(nm_mapper_hidden_size)
        nm_cfg["output_mlp_hidden"] = int(nm_mapper_hidden_size)
        nm_cfg["mlp_hidden"] = int(nm_mapper_hidden_size)
        nm_cfg["mapper_hidden_exact"] = True
    settings["nm_cfg"] = nm_cfg
    settings["mod_mask_cfg"] = {
        "fixed_enable": bool(args.mod_fixed_mask_enable),
        "fixed_seed": args.mod_fixed_mask_seed,
        "seed": args.seed,
        "fixed_flat_inputs": bool(args.mod_fixed_mask_flat_inputs),
        "zero_fallback": bool(args.mod_fixed_mask_zero_fallback),
    }
    settings["mod_hid_flat_group"] = bool(args.mod_hid_flat_group)
    settings["mod_hid_flat_modulated_only"] = bool(args.mod_hid_flat_modulated_only)

    settings["param_timescales"] = {
        "enable": args.param_timescales_enable,
        "distribution": args.param_timescales_distribution,
        "scale": args.param_timescales_scale,
        "std": args.param_timescales_std,
        "seed": args.param_timescales_seed,
        "trainable": args.param_timescales_trainable,
    }

    smoothing_modes_raw = args.param_smoothing_modes or []
    smoothing_modes: List[str] = []
    for mode in smoothing_modes_raw:
        lowered = str(mode).lower()
        if lowered == "all":
            smoothing_modes = ["mlp_sub", "mlp_add", "snn_add"]
            break
        smoothing_modes.append(_normalize_mlp_mode(lowered))
    settings["param_smoothing"] = {
        "enable": args.param_smoothing_enable,
        "tau_init": args.param_smoothing_tau_init,
        "tau_min": args.param_smoothing_tau_min,
        "trainable": args.param_smoothing_trainable,
        "modes": smoothing_modes,
    }

    # split / training controls
    use_validation   = args.use_validation
    val_fraction     = args.val_fraction
    k_folds          = args.k_folds
    patience         = args.patience
    seed             = args.seed

    def _normalize_optional_path(value):
        if value is None:
            return None
        value_str = str(value).strip()
        if not value_str or value_str.lower() == "none":
            return None
        return value_str

    val_file = _normalize_optional_path(args.val_file)
    settings["val_file"] = val_file

    fixed_split_path = _normalize_optional_path(args.fixed_split_path)
    reinit_per_fold  = args.reinit_per_fold

    # staged controls
    staged_unlock_metric    = args.staged_unlock_metric
    staged_unlock_threshold = args.staged_unlock_threshold
    staged_min_epochs       = args.staged_min_epochs  # currently not used explicitly; reserved

    # mod-only controls
    start_locked   = args.start_locked
    unlock_metric  = args.unlock_metric
    unlock_threshold = args.unlock_threshold

    # base ckpt
    base_snn_ckpt = _normalize_optional_path(args.base_snn_ckpt)
    base_snn_stockpile_dir = _normalize_optional_path(args.base_snn_stockpile_dir)
    use_stockpile_base = bool(args.base_snn_from_stockpile)
    if use_stockpile_base and base_snn_ckpt is not None:
        print("[info] --base_snn_from_stockpile enabled; ignoring --base_snn_ckpt.")
        base_snn_ckpt = None
    if use_stockpile_base and run_mode != "mod":
        print("[info] --base_snn_from_stockpile applies only to run_mode='mod'; ignoring for this run.")
        use_stockpile_base = False
    snn_save_every_epoch = args.save_every_epoch
    mod_save_every_epoch = False
    if args.save_every_epoch and run_mode in ("mod", "staged"):
        print("[info] --save_every_epoch applies only to the SNN phase; "
              "mod checkpoints are saved only when they improve (mod_best.pth).")

    # determine global run index (align SNN and Mod directories)
    aligned_run_index = next_aligned_run_index(settings["save_dir_snn"], settings["save_dir_mod"])

    # choose save dir by mode
    if run_mode == "snn":
        settings["save_dir"] = settings["save_dir_snn"]
    elif run_mode in ("mod", "staged"):
        settings["save_dir"] = settings["save_dir_mod"]

    # load data
    x_train, y_train, x_test, y_test = load_h5(
        cache_dir=settings["cache_dir"], cache_subdir=settings["cache_subdir"],
        train_file=settings["train_file"], test_file=settings["test_file"]
    )
    x_val, y_val = (None, None)
    if val_file:
        x_val, y_val = load_validation_split(
            cache_dir=settings["cache_dir"], cache_subdir=settings["cache_subdir"], val_file=val_file
        )

    # build base state
    state = setup_model(settings)

    # === Execute ===
    if run_mode == "snn":
        print(format_param_stats(state, None, prefix="Run mode=snn:"))
        # use settings["nb_epochs"] directly
        result = train_snn_hetero(
            state, settings, x_train, y_train, x_test, y_test, x_val=x_val, y_val=y_val,
            save_every_epoch=snn_save_every_epoch,
            use_validation=args.use_validation, val_fraction=args.val_fraction,
            k_folds=args.k_folds, patience=args.patience, seed=args.seed,
            fixed_split_path=fixed_split_path, reinit_per_fold=args.reinit_per_fold,
            chosen_metric=args.staged_unlock_metric,
            chosen_threshold=args.staged_unlock_threshold,
            run_index=aligned_run_index,
        )

        print("\nSNN training complete.")
        print(result)

    elif run_mode == "mod":
        # override epochs for mod if provided
        settings_mod = dict(settings)
        settings_mod["nb_epochs"] = args.nb_epochs_mod if args.nb_epochs_mod is not None else args.nb_epochs
        stockpile_ckpt = None
        stockpile_split_path = None
        if use_stockpile_base:
            stockpile_source = base_snn_stockpile_dir or "SNN_Stockpile/Base_SNN/SNN"
            stockpile_ckpt = pick_random_stockpile_ckpt(stockpile_source, seed=args.base_snn_stockpile_seed)
            candidate_split = stockpile_ckpt.parent / "split_indices.npz"
            if candidate_split.exists():
                stockpile_split_path = str(candidate_split)
                print(f"[info] Using stockpile split indices from {candidate_split}")
            else:
                print(f"[warn] Stockpile split indices not found next to {stockpile_ckpt}; using default split settings.")
        effective_base_ckpt = stockpile_ckpt or base_snn_ckpt
        effective_fixed_split = stockpile_split_path or fixed_split_path
        use_validation_mod = args.use_validation or (stockpile_split_path is not None)
        if (stockpile_split_path is not None) and (not args.use_validation):
            print("[info] Enabling validation to reuse stockpile split indices.")
        if effective_base_ckpt is not None:
            state_base = load_base_snn_state(effective_base_ckpt, settings_mod.get("snn_train_flags"), settings=settings_mod)
        else:
            state_base = setup_model(settings_mod)

        mod_mode = _select_mlp_mode(settings_mod)
        mlp_hidden_sizes_mod = _resolve_mlp_hidden_sizes(settings_mod)
        snn_hidden_sizes_mod = settings_mod.get("mod_hidden_sizes")
        hidden_cfg = snn_hidden_sizes_mod if _is_snn_mode(mod_mode) else mlp_hidden_sizes_mod
        modulator = build_modulator(
            settings_mod,
            override_mode=mod_mode,
            hidden_sizes=hidden_cfg,
        )
        print(format_param_stats(state_base, modulator, prefix="Run mode=mod:", settings=settings_mod))
        result = train_modulated_snn(
            state_base, settings_mod, x_train, y_train, x_test, y_test, x_val=x_val, y_val=y_val,
            modulator=modulator, mlp_interval=settings_mod["mlp_interval"],
            save_every_epoch=mod_save_every_epoch, use_validation=use_validation_mod, val_fraction=args.val_fraction,
            k_folds=args.k_folds, patience=args.patience, seed=args.seed,
            fixed_split_path=effective_fixed_split, reinit_per_fold=args.reinit_per_fold,
            start_locked=args.start_locked, unlock_metric=args.unlock_metric, unlock_threshold=args.unlock_threshold,
            base_ckpt_path=effective_base_ckpt,  # <— NEW
            run_index=aligned_run_index,
        )


        print("\nModulated training complete.")
        print(result)
        maybe_run_ann_shap(result, settings_mod, shap_cfg, x_train, y_train, x_test, y_test)

    elif run_mode == "staged":
        # ---- Stage 1: SNN ----
        settings_stage1 = dict(settings)
        settings_stage1["save_dir"] = settings["save_dir_snn"]
        settings_stage1["nb_epochs"] = args.nb_epochs_snn

        print(format_param_stats(state, None, prefix="Stage-1 snn:"))
        result = train_snn_hetero(
            state, settings_stage1, x_train, y_train, x_test, y_test, x_val=x_val, y_val=y_val,
            save_every_epoch=snn_save_every_epoch, use_validation=args.use_validation, val_fraction=args.val_fraction,
            k_folds=args.k_folds, patience=args.patience, seed=args.seed,
            fixed_split_path=fixed_split_path, reinit_per_fold=args.reinit_per_fold,
            chosen_metric=args.staged_unlock_metric,
            chosen_threshold=args.staged_unlock_threshold,
            run_index=aligned_run_index,
        )

        print("\nStage-1 (SNN) complete.")
        print(result)

        stage_run_index = _parse_run_index(Path(result["run_dir"]).name)
        if stage_run_index is None:
            stage_run_index = aligned_run_index

        # pick 'chosen' for Stage-2; fallback to snn_best.pth if needed
        fold1_dir = Path(result["run_dir"]) / "Fold_1"
        chosen_candidates = [fold1_dir / "chosen.pth", fold1_dir / "chosen"]
        chosen_ckpt = next((p for p in chosen_candidates if p.exists()), None)
        if chosen_ckpt is None:
            fallback_candidates = [fold1_dir / "snn_best.pth", fold1_dir / "best_snn.pth"]
            for cand in fallback_candidates:
                if cand.exists():
                    chosen_ckpt = cand
                    print(f"⚠️ 'chosen' not found; using fallback checkpoint {cand}")
                    break
        if chosen_ckpt is None:
            raise FileNotFoundError(f"No base SNN checkpoint found in {fold1_dir}")
        print(f"[Stage-2] Using base SNN checkpoint: {chosen_ckpt}")

        # ---- Stage 2: Modulated ----
        settings_mod = dict(settings)
        settings_mod["nb_epochs"] = args.nb_epochs_mod
        state_base = load_base_snn_state(chosen_ckpt, settings_mod.get("snn_train_flags"), settings=settings_mod)
        mod_mode = _select_mlp_mode(settings_mod)
        mlp_hidden_sizes_stage2 = _resolve_mlp_hidden_sizes(settings_mod)
        snn_hidden_sizes_stage2 = settings_mod.get("mod_hidden_sizes")
        hidden_cfg = snn_hidden_sizes_stage2 if _is_snn_mode(mod_mode) else mlp_hidden_sizes_stage2
        modulator = build_modulator(
            settings_mod,
            override_mode=mod_mode,
            hidden_sizes=hidden_cfg,
        )
        print(format_param_stats(state_base, modulator, prefix="Stage-2 mod:", settings=settings_mod))

        result_mod = train_modulated_snn(
            state_base, settings_mod, x_train, y_train, x_test, y_test, x_val=x_val, y_val=y_val,
            modulator=modulator, mlp_interval=settings_mod["mlp_interval"],
            save_every_epoch=mod_save_every_epoch, use_validation=args.use_validation, val_fraction=args.val_fraction,
            k_folds=args.k_folds, patience=args.patience, seed=args.seed,
            fixed_split_path=fixed_split_path, reinit_per_fold=args.reinit_per_fold,
            start_locked=args.start_locked,                     
            unlock_metric=args.unlock_metric,                    
            unlock_threshold=args.unlock_threshold,              
            base_ckpt_path=chosen_ckpt,
            run_index=stage_run_index,
        )

        print("\nStage-2 (Modulated) complete.")
        print(result_mod)
        maybe_run_ann_shap(result_mod, settings_mod, shap_cfg, x_train, y_train, x_test, y_test)
    else:
        raise ValueError(f"Unknown run_mode: {run_mode}")


if __name__ == "__main__":
    main()
