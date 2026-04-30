"""Data loading, spike augmentation, batching, and split helpers."""
from .utils import *

def jitter_times(times, sigma_ms, max_time):
    if sigma_ms <= 0: return times
    noise = np.random.normal(0.0, sigma_ms/1000.0, size=times.shape)
    t = times + noise
    return np.clip(t, 0.0, max_time)

def shift_times(times, shift_ms, max_time):
    if shift_ms == 0: return times
    return np.clip(times + (shift_ms/1000.0), 0.0, max_time)

def scale_times(times, scale, max_time):
    return np.clip(times * scale, 0.0, max_time)

def event_drop(times, units, drop_p):
    if drop_p <= 0: return times, units
    keep = np.random.rand(times.shape[0]) > drop_p
    return times[keep], units[keep]

def event_insert(times, units, nb_units, rate=0.0, max_time=1.4):
    n_new = int(len(times) * rate)
    if n_new <= 0: return times, units
    new_t = np.random.rand(n_new) * max_time
    new_u = np.random.randint(0, nb_units, size=n_new)
    return np.concatenate([times, new_t]), np.concatenate([units, new_u])

def band_mask(times, units, nb_units, band_frac=0.0):
    if band_frac <= 0: return times, units
    band = max(1, int(nb_units * band_frac))
    start = np.random.randint(0, nb_units - band + 1)
    keep = (units < start) | (units >= start + band)
    return times[keep], units[keep]

def compress_units(units, factor: int, nb_units_new: int):
    if factor <= 1: return units
    u = (units // factor).astype(int)
    return np.clip(u, 0, nb_units_new - 1)

def compress_dense_inputs(inputs: torch.Tensor, factor: int, nb_units_new: Optional[int] = None) -> torch.Tensor:
    if factor <= 1:
        return inputs
    B, T, C = inputs.shape
    if nb_units_new is None:
        nb_units_new = int(math.ceil(C / float(factor)))
    pad = nb_units_new * factor - C
    if pad > 0:
        inputs = F.pad(inputs, (0, pad))
    reshaped = inputs.reshape(B, T, nb_units_new, factor)
    return reshaped.sum(dim=3)

def channel_jitter(units, nb_units, sigma_units=0.0):
    if sigma_units <= 0: return units
    jitter = np.random.normal(0.0, sigma_units, size=units.shape)
    u = np.round(units + jitter).astype(int)
    return np.clip(u, 0, nb_units - 1)

def inject_poisson_noise(times, units, nb_units, rate_hz=0.0, max_time=1.4, per_input: bool = False):
    if rate_hz <= 0: return times, units
    total_rate = rate_hz * (nb_units if per_input else 1.0)
    n_new = np.random.poisson(total_rate * max_time)
    if n_new <= 0: return times, units
    new_t = np.random.rand(n_new) * max_time
    new_u = np.random.randint(0, nb_units, size=n_new)
    return np.concatenate([times, new_t]), np.concatenate([units, new_u])

def time_mask_postbin(X_dense, mask_frac=0.0):
    if mask_frac <= 0: return X_dense
    T = X_dense.shape[1]
    L = max(1, int(T * mask_frac))
    s = np.random.randint(0, T - L + 1)
    X_dense[:, s:s+L, :] = 0.0
    return X_dense

def augment_spike_train(times, units, cfg, nb_units, max_time):
    compress_factor = int(cfg.get("compress_factor", 1))
    jitter_std = cfg.get("channel_jitter_std_eff")
    if jitter_std is None:
        jitter_std = cfg.get("channel_jitter_std", 0.0)
        if compress_factor > 1 and jitter_std > 0:
            jitter_std = jitter_std / float(compress_factor)
    noise_rate_hz = cfg.get("noise_rate_hz_eff")
    if noise_rate_hz is None:
        noise_rate_hz = cfg.get("noise_rate_hz", 0.0)
        if compress_factor > 1 and cfg.get("noise_per_input", False):
            noise_rate_hz = noise_rate_hz * float(compress_factor)
    if compress_factor > 1:
        units = compress_units(units, compress_factor, nb_units)
    if cfg.get("jitter_ms", 0) > 0:
        times = jitter_times(times, np.random.uniform(0, cfg["jitter_ms"]), max_time)
    if cfg.get("shift_ms", 0) > 0:
        s = np.random.uniform(-cfg["shift_ms"], cfg["shift_ms"])
        times = shift_times(times, s, max_time)
    if cfg.get("scale_low", 1.0) != 1.0 or cfg.get("scale_high", 1.0) != 1.0:
        scale = np.random.uniform(cfg.get("scale_low", 1.0), cfg.get("scale_high", 1.0))
        times = scale_times(times, scale, max_time)
    if jitter_std > 0:
        units = channel_jitter(units, nb_units, jitter_std)
    if cfg.get("drop_p", 0) > 0:
        times, units = event_drop(times, units, cfg["drop_p"])
    if cfg.get("insert_rate", 0) > 0:
        times, units = event_insert(times, units, nb_units, cfg["insert_rate"], max_time)
    if noise_rate_hz > 0:
        times, units = inject_poisson_noise(
            times, units, nb_units,
            rate_hz=noise_rate_hz,
            max_time=max_time,
            per_input=cfg.get("noise_per_input", False)
        )
    if cfg.get("band_frac", 0) > 0:
        times, units = band_mask(times, units, nb_units, cfg["band_frac"])
    return times, units

# -------------------------
# Data generator (with optional subset indices + augmentation)
# -------------------------
def sparse_data_generator_from_hdf5_spikes(
    X, y, batch_size, nb_steps, nb_units, max_time, shuffle=True, indices=None,
    augment_cfg: dict=None, postbin_time_mask: float=0.0
):
    labels_full = np.asarray(y, dtype=int)
    sample_index = np.arange(len(labels_full)) if indices is None else np.asarray(indices)
    number_of_batches = len(sample_index) // batch_size
    if shuffle: np.random.shuffle(sample_index)
    firing_times, units_fired = X['times'], X['units']
    time_bins = np.linspace(0, max_time, num=nb_steps)

    counter = 0
    while counter < number_of_batches:
        batch_index = sample_index[batch_size*counter: batch_size*(counter+1)]
        coo = [[],[],[]]
        for bc, idx in enumerate(batch_index):
            times = np.array(firing_times[idx]); units = np.array(units_fired[idx])
            if augment_cfg:
                times, units = augment_spike_train(times, units, augment_cfg, nb_units, max_time)
            bins = np.digitize(times, time_bins)
            coo[0].extend([bc]*len(bins)); coo[1].extend(bins); coo[2].extend(units)
        i = torch.LongTensor(coo).to(device)
        v = torch.FloatTensor(np.ones(len(coo[0]))).to(device)
        X_batch = torch.sparse_coo_tensor(
            i, v, torch.Size([len(batch_index), nb_steps, nb_units]),
            dtype=dtype, device=device
        )
        if postbin_time_mask > 0.0:
            Xd = X_batch.to_dense()
            Xd = torch.from_numpy(time_mask_postbin(Xd.cpu().numpy(), postbin_time_mask)).to(device)
            X_batch = Xd.to_sparse()
        y_batch = torch.tensor(labels_full[batch_index], device=device)
        yield X_batch, y_batch
        counter += 1

# -------------------------
# Split helpers + persistence
# -------------------------
def stratified_split_indices(y, val_fraction=0.1, seed=0):
    y_np = np.array(y, dtype=int)
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for c in np.unique(y_np):
        idx_c = np.where(y_np == c)[0]
        rng.shuffle(idx_c)
        n_val = max(1, int(len(idx_c) * val_fraction))
        val_idx.extend(idx_c[:n_val]); train_idx.extend(idx_c[n_val:])
    return np.array(train_idx), np.array(val_idx)

def make_kfold_splits(y, k_folds=5, seed=0):
    y_np = np.array(y, dtype=int)
    rng = np.random.default_rng(seed)
    per_class = {}
    for c in np.unique(y_np):
        idx_c = np.where(y_np == c)[0]
        rng.shuffle(idx_c)
        per_class[c] = np.array_split(idx_c, k_folds)
    folds = []
    for k in range(k_folds):
        val_idx = np.concatenate([per_class[c][k] for c in per_class])
        train_idx = np.concatenate([np.concatenate([per_class[c][i] for i in range(k_folds) if i != k]) for c in per_class])
        folds.append((train_idx, val_idx))
    return folds

def save_indices(path: Path, train_idx: np.ndarray, val_idx: Optional[np.ndarray], meta: dict = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, train_idx=train_idx,
                        val_idx=val_idx if val_idx is not None else np.array([], dtype=int),
                        meta=meta or {})

def load_indices(path: Path):
    data = np.load(path, allow_pickle=True)
    train_idx = data["train_idx"]
    val_idx_arr = data["val_idx"]
    val_idx = val_idx_arr if val_idx_arr.size > 0 else None
    meta = dict(data["meta"].item()) if "meta" in data else {}
    return train_idx, val_idx, meta

# -------------------------
# Surrogate gradient spike
# -------------------------


__all__ = [name for name in globals() if not name.startswith('__')]
