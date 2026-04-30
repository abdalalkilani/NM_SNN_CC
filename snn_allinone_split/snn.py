"""Base SNN model, forward pass, evaluation, and training."""
from .data import *
from .grouping import *
from .reporting import *

class SurrGradSpike(torch.autograd.Function):
    scale = 100.0
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out
    @staticmethod
    def backward(ctx, grad_output):
        (inp,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input / (SurrGradSpike.scale * torch.abs(inp) + 1.0) ** 2

spike_fn = SurrGradSpike.apply

# -------------------------
# Tau/time-step helpers
# -------------------------
def _apply_input_delay(
    inputs_dense: torch.Tensor,
    delay_logits: Optional[torch.Tensor],
    max_delay: int,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Apply a learnable axonal delay per input channel by shifting spike trains.
    Straight-through argmax over delays 0..max_delay: forward uses one-hot,
    backward flows through softmax for gradients.
    """
    if delay_logits is None or max_delay <= 0:
        return inputs_dense
    B, T, C = inputs_dense.shape
    max_delay = min(max_delay, delay_logits.size(1) - 1)
    logits = delay_logits[:, : max_delay + 1].to(device=inputs_dense.device, dtype=inputs_dense.dtype)  # [C,K]
    temp = max(1e-3, float(temperature))
    probs = F.softmax(logits / temp, dim=1)
    hard_idx = probs.argmax(dim=1)
    hard = F.one_hot(hard_idx, num_classes=probs.size(1)).to(device=inputs_dense.device, dtype=inputs_dense.dtype)
    weights = hard + (probs - probs.detach())  # straight-through
    out = torch.zeros_like(inputs_dense)
    for k in range(max_delay + 1):
        if k >= T:
            break
        shifted = torch.cat(
            [torch.zeros((B, k, C), device=inputs_dense.device, dtype=inputs_dense.dtype), inputs_dense[:, : T - k, :]],
            dim=1,
        )
        out = out + shifted * weights[:, k].view(1, 1, C)
    return out

# -------------------------
# Base SNN state + forward
# -------------------------
def setup_model(settings: Dict):
    nb_inputs   = settings["nb_inputs"]
    nb_hidden   = settings["nb_hidden"]
    nb_outputs  = settings["nb_outputs"]
    time_step   = settings["time_step"]
    tau_syn     = settings["tau_syn"]
    tau_mem     = settings["tau_mem"]
    weight_scale= settings["weight_scale"]

    # Optionally align decay step with clip duration so taus stay in real-time units
    decay_time_step = _effective_tau_decay_dt(settings)

    param_ranges = _resolve_param_ranges(settings)
    thr_lo, thr_hi = param_ranges["thr"]
    reset_lo, reset_hi = param_ranges["reset"]
    rest_lo, rest_hi = param_ranges["rest"]
    thresholds_1 = torch.empty((1, nb_hidden), device=device, dtype=dtype, requires_grad=True)
    torch.nn.init.uniform_(thresholds_1, a=float(thr_lo), b=float(thr_hi))
    reset_1 = torch.empty((1, nb_hidden), device=device, dtype=dtype, requires_grad=True)
    torch.nn.init.uniform_(reset_1, a=float(reset_lo), b=float(reset_hi))
    rest_1 = torch.empty((1, nb_hidden), device=device, dtype=dtype, requires_grad=True)
    torch.nn.init.uniform_(rest_1, a=float(rest_lo), b=float(rest_hi))

    homo = bool(settings.get("homo_init", False))
    if homo:
        thr_val = 1.0
        reset_val = 0.0
        rest_val = 0.0
        thresholds_1.data.fill_(thr_val)
        reset_1.data.fill_(reset_val)
        rest_1.data.fill_(rest_val)

        alpha_val = math.exp(-decay_time_step / tau_syn)
        beta_val = math.exp(-decay_time_step / tau_mem)
        alpha_hetero_1 = torch.full((1, nb_hidden), alpha_val, device=device, dtype=dtype, requires_grad=True)
        beta_hetero_1  = torch.full((1, nb_hidden), beta_val, device=device, dtype=dtype, requires_grad=True)
        alpha_hetero_2 = torch.full((1, nb_outputs), alpha_val, device=device, dtype=dtype, requires_grad=True)
        beta_hetero_2  = torch.full((1, nb_outputs), beta_val, device=device, dtype=dtype, requires_grad=True)
    else:
        distribution = dist_fn('gamma')
        alpha_hetero_1_dist = torch.tensor(distribution(tau_syn, 3, (1, nb_hidden)), device=device, dtype=dtype)
        alpha_hetero_1 = torch.exp(-decay_time_step / alpha_hetero_1_dist).requires_grad_(True)
        beta_hetero_1_dist = torch.tensor(distribution(tau_mem, 3, (1, nb_hidden)), device=device, dtype=dtype)
        beta_hetero_1 = torch.exp(-decay_time_step / beta_hetero_1_dist).requires_grad_(True)
        alpha_hetero_2_dist = torch.tensor(distribution(tau_syn, 3, (1, nb_outputs)), device=device, dtype=dtype)
        alpha_hetero_2 = torch.exp(-decay_time_step / alpha_hetero_2_dist).requires_grad_(True)
        beta_hetero_2_dist = torch.tensor(distribution(tau_mem, 3, (1, nb_outputs)), device=device, dtype=dtype)
        beta_hetero_2 = torch.exp(-decay_time_step / beta_hetero_2_dist).requires_grad_(True)
        with torch.no_grad():
            alpha_hetero_1.clamp_(float(param_ranges["alpha_1"][0]), float(param_ranges["alpha_1"][1]))
            beta_hetero_1.clamp_(float(param_ranges["beta_1"][0]), float(param_ranges["beta_1"][1]))
            alpha_hetero_2.clamp_(float(param_ranges["alpha_2"][0]), float(param_ranges["alpha_2"][1]))
            beta_hetero_2.clamp_(float(param_ranges["beta_2"][0]), float(param_ranges["beta_2"][1]))

    # w1 = torch.empty((nb_inputs, nb_hidden), device=device, dtype=dtype, requires_grad=True)
    # torch.nn.init.normal_(w1, mean=0.0, std=weight_scale / math.sqrt(nb_inputs))
    # w2 = torch.empty((nb_hidden, nb_outputs), device=device, dtype=dtype, requires_grad=True)
    # torch.nn.init.normal_(w2, mean=0.0, std=weight_scale / math.sqrt(nb_hidden))
    # v1 = torch.empty((nb_hidden, nb_hidden), device=device, dtype=dtype, requires_grad=True)
    # torch.nn.init.normal_(v1, mean=0.0, std=weight_scale / math.sqrt(nb_hidden))

    # ---- weight initialization ----

    # w1 ~ Normal(-0.00046, 0.0215²)
    w1 = torch.empty((nb_inputs, nb_hidden), device=device, dtype=dtype, requires_grad=True)
    torch.nn.init.normal_(w1, mean=-0.00046421893, std=0.021497283)

    # v1 ~ Normal(-0.00443, 0.0257²)
    v1 = torch.empty((nb_hidden, nb_hidden), device=device, dtype=dtype, requires_grad=True)
    torch.nn.init.normal_(v1, mean=-0.004429796, std=0.025746185)

    # w2 ~ mixture of 2 Gaussians
    weights = [0.61827673, 0.38172327]
    means   = [0.01217757, -0.03004124]
    stds    = [0.03018986, 0.04652500]

    comp_mask = torch.rand((nb_hidden, nb_outputs), device=device, dtype=dtype) < weights[0]
    z = torch.randn((nb_hidden, nb_outputs), device=device, dtype=dtype)
    w2 = torch.where(
        comp_mask,
        means[0] + stds[0] * z,
        means[1] + stds[1] * z
    ).requires_grad_()

    # learnable input delay kernels (per-input softmax over [0, max_delay])
    delay_logits = _init_input_delay_logits(settings, nb_inputs) if settings.get("use_input_delay", False) else None

    # --------------------------------

    state = dict(
        thresholds_1=thresholds_1, reset_1=reset_1, rest_1=rest_1,
        alpha_hetero_1=alpha_hetero_1, beta_hetero_1=beta_hetero_1,
        alpha_hetero_2=alpha_hetero_2, beta_hetero_2=beta_hetero_2,
        w1=w1, w2=w2, v1=v1,
    )
    if delay_logits is not None:
        state["input_delay_logits"] = delay_logits
    return _apply_snn_train_flags(state, _snn_train_flags(settings))

def run_snn_hetero(inputs_dense, state: Dict, settings: Dict):
    nb_hidden   = settings["nb_hidden"]
    nb_outputs  = settings["nb_outputs"]
    nb_steps    = settings["nb_steps"]
    batch_size  = settings["batch_size"]
    training = bool(settings.get("training", False))
    hidden_dropout_p = float(settings.get("hidden_dropout_p", 0.0))
    psp_norm = bool(settings.get("psp_norm_peak", False))

    w1, w2, v1 = state["w1"], state["w2"], state["v1"]
    alpha_hetero_1 = state["alpha_hetero_1"]
    beta_hetero_1  = state["beta_hetero_1"]
    alpha_hetero_2 = state["alpha_hetero_2"]
    beta_hetero_2  = state["beta_hetero_2"]
    thresholds_1   = state["thresholds_1"]
    reset_1        = state["reset_1"]
    rest_1         = state["rest_1"]

    syn = torch.zeros((batch_size, nb_hidden), device=device, dtype=dtype)
    mem = torch.zeros((batch_size, nb_hidden), device=device, dtype=dtype)
    decay_dt = _effective_tau_decay_dt(settings)
    gain_hidden = None
    gain_out = None
    if psp_norm:
        gain_hidden = _psp_peak_gain(alpha_hetero_1, beta_hetero_1, dt=decay_dt)
        gain_out = _psp_peak_gain(alpha_hetero_2, beta_hetero_2, dt=decay_dt)

    mem_rec, spk_rec = [], []

    # feedforward input current sequence to hidden: [B,T,H]
    delay_logits = state.get("input_delay_logits")
    max_delay = max(1, int(settings.get("input_delay_steps", 1))) if settings.get("use_input_delay", False) else 0
    inputs_delayed = _apply_input_delay(
        inputs_dense, delay_logits, max_delay, temperature=settings.get("input_delay_temp", 1.0)
    )
    h1_from_input = torch.einsum("btc,cd->btd", inputs_delayed, w1)

    out = torch.zeros((batch_size, nb_hidden), device=device, dtype=dtype)

    for t in range(nb_steps):
        # total hidden current at time t (input + recurrent)
        h1_t = h1_from_input[:, t] + torch.einsum("bd,dc->bc", out, v1)  # [B,H]

        # SNN update
        mthr = mem - thresholds_1
        spk  = spike_fn(mthr)
        if hidden_dropout_p > 0.0 and training:
            spk = F.dropout(spk, p=hidden_dropout_p, training=True)
        rst  = (mthr > 0).float()

        syn = alpha_hetero_1 * syn + h1_t
        syn_term = (1 - beta_hetero_1) * syn
        if gain_hidden is not None:
            syn_term = gain_hidden * syn_term
        mem = beta_hetero_1 * (mem - rest_1) + rest_1 + syn_term - rst * (thresholds_1 - reset_1)

        mem_rec.append(mem)
        spk_rec.append(spk)
        out = spk

    mem_rec = torch.stack(mem_rec, dim=1)   # [B,T,H]
    spk_rec = torch.stack(spk_rec, dim=1)   # [B,T,H]

    # raw output current from hidden spikes (before output filters)
    h2 = torch.einsum("bth,ho->bto", spk_rec, w2)  # [B,T,O]

    flt = torch.zeros((batch_size, nb_outputs), device=device, dtype=dtype)
    out = torch.zeros((batch_size, nb_outputs), device=device, dtype=dtype)
    out_rec = [out]
    for t in range(nb_steps):
        flt = alpha_hetero_2 * flt + h2[:, t]
        flt_term = (1 - beta_hetero_2) * flt
        if gain_out is not None:
            flt_term = gain_out * flt_term
        out = beta_hetero_2 * out + flt_term
        out_rec.append(out)
    out_rec = torch.stack(out_rec, dim=1)
    return out_rec, (mem_rec, spk_rec)

# -------------------------
# Eval helpers (SNN)
# -------------------------
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
    # Imported here to avoid a module loop at package import time.
    from .modulated import load_base_snn_state, compute_snn_test_spike_stats, _format_spike_stats

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
        input_delay_steps=settings.get("input_delay_steps"),
        use_input_delay=settings.get("use_input_delay"),
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

# -------------------------
# MLP (configurable IO groups)
# -------------------------


__all__ = [name for name in globals() if not name.startswith('__')]
