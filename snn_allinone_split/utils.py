"""General helpers shared across the runner."""
from .config import *

def _resolve_h5_path(cache_dir: Union[str, Path], cache_subdir: Optional[Union[str, Path]], file_name: str) -> str:
    if os.path.isabs(file_name):
        return file_name
    base = Path(os.path.expanduser(str(cache_dir or ".")))
    if cache_subdir:
        base = base / str(cache_subdir)
    return str(base / file_name)

def _normalize_train_flags(flags: Optional[Dict[str, bool]]) -> Dict[str, bool]:
    return {name: bool((flags or {}).get(name, True)) for name in SNN_PARAM_NAMES}

def _snn_train_flags(settings: Dict[str, Union[bool, Dict[str, bool]]]) -> Dict[str, bool]:
    return _normalize_train_flags(settings.get("snn_train_flags"))

def _apply_snn_train_flags(state: Dict[str, torch.nn.Parameter], flags: Dict[str, bool]):
    for name, param in state.items():
        if isinstance(param, torch.nn.Parameter):
            param.requires_grad_(flags.get(name, True))
    return state

def _trainable_params_from(state: Dict[str, torch.nn.Parameter], names: Optional[List[str]] = None) -> List[torch.nn.Parameter]:
    names = names or SNN_PARAM_NAMES
    return [state[n] for n in names if state.get(n) is not None and state[n].requires_grad]

# -------------------------
# Utilities
# -------------------------
def get_run_dir(base_dir: Union[str, Path]) -> Path:
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    i = 1
    while True:
        d = base / f"Run_{i}"
        try:
            d.mkdir(parents=True, exist_ok=False)
            return d
        except FileExistsError:
            i += 1


def _parse_run_index(name: str) -> Optional[int]:
    if not name.startswith("Run_"):
        return None
    try:
        return int(name.split("_", 1)[1])
    except (IndexError, ValueError):
        return None


def _existing_run_indices(base_dir: Union[str, Path]) -> List[int]:
    base = Path(base_dir)
    if not base.exists():
        return []
    indices = []
    for child in base.iterdir():
        if child.is_dir():
            idx = _parse_run_index(child.name)
            if idx is not None:
                indices.append(idx)
    return sorted(indices)


def next_aligned_run_index(*base_dirs: Union[str, Path, None]) -> int:
    max_idx = 0
    for base in base_dirs:
        if base is None:
            continue
        indices = _existing_run_indices(base)
        if indices:
            max_idx = max(max_idx, indices[-1])
    return max_idx + 1


def ensure_run_dir(base_dir: Union[str, Path], run_index: Optional[int] = None) -> Path:
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    if run_index is None:
        return get_run_dir(base)
    next_index = int(run_index)
    while True:
        run_dir = base / f"Run_{next_index}"
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            return run_dir
        except FileExistsError:
            next_index += 1

def dist_fn(dist):
    return {
        'gamma': lambda mean, k, size: np.random.gamma(k, scale=mean/k, size=size),
        'normal': lambda mean, k, size: np.random.normal(loc=mean, scale=mean/np.sqrt(k), size=size),
        'uniform': lambda _, maximum, size: np.random.uniform(low=0, high=maximum, size=size),
    }[dist.lower()]

def load_h5(cache_dir: str, cache_subdir: str, train_file: str, test_file: str):
    train_path = _resolve_h5_path(cache_dir, cache_subdir, train_file)
    test_path  = _resolve_h5_path(cache_dir, cache_subdir, test_file)
    train_h5 = h5py.File(train_path, 'r'); test_h5  = h5py.File(test_path, 'r')
    return train_h5['spikes'], train_h5['labels'], test_h5['spikes'], test_h5['labels']


def load_validation_split(cache_dir: str, cache_subdir: str, val_file: Optional[str]):
    if not val_file:
        return None, None
    val_path = _resolve_h5_path(cache_dir, cache_subdir, val_file)
    val_h5 = h5py.File(val_path, 'r')
    return val_h5['spikes'], val_h5['labels']

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_int_list(arg: Optional[Union[str, List[int], Tuple[int, ...]]]) -> Optional[List[int]]:
    if arg is None:
        return None
    if isinstance(arg, (list, tuple)):
        return [int(v) for v in arg]
    text = str(arg).strip()
    if not text:
        return None
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)):
            return [int(v) for v in parsed]
    except (ValueError, SyntaxError):
        pass
    tokens = [tok for tok in text.replace(",", " ").split() if tok]
    if not tokens:
        return None
    return [int(tok) for tok in tokens]


def parse_float_list(arg: Optional[Union[str, List[float], Tuple[float, ...]]]) -> Optional[List[float]]:
    if arg is None:
        return None
    if isinstance(arg, (list, tuple)):
        return [float(v) for v in arg]
    text = str(arg).strip()
    if not text:
        return None
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)):
            return [float(v) for v in parsed]
    except (ValueError, SyntaxError):
        pass
    tokens = [tok for tok in text.replace(",", " ").split() if tok]
    if not tokens:
        return None
    return [float(tok) for tok in tokens]


def parse_str_list(arg: Optional[Union[str, List[str], Tuple[str, ...]]]) -> Optional[List[str]]:
    if arg is None:
        return None
    if isinstance(arg, (list, tuple)):
        if len(arg) == 1 and isinstance(arg[0], str):
            return parse_str_list(arg[0])
        return [str(v) for v in arg]
    text = str(arg).strip()
    if not text:
        return None
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)):
            return [str(v) for v in parsed]
    except (ValueError, SyntaxError):
        pass
    tokens = [tok for tok in text.replace(",", " ").split() if tok]
    if not tokens:
        return None
    return [str(tok) for tok in tokens]


def _normalize_combo_param_names(names: Optional[List[str]]) -> List[str]:
    if not names:
        return []
    normalized = []
    aliases = {
        "threshold": "thr",
        "thresholds": "thr",
        "thresh": "thr",
    }
    for name in names:
        if name is None:
            continue
        key = str(name).strip().lower()
        if not key:
            continue
        key = key.replace("-", "_")
        key = key.replace(" ", "")
        key = aliases.get(key, key)
        if key in {"alpha1", "alpha_1"}:
            key = "alpha_1"
        elif key in {"beta1", "beta_1"}:
            key = "beta_1"
        elif key in {"alpha2", "alpha_2"}:
            key = "alpha_2"
        elif key in {"beta2", "beta_2"}:
            key = "beta_2"
        if key in MOD_TARGET_PARAM_NAMES and key not in normalized:
            normalized.append(key)
    return normalized


def _normalize_param_name(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    normalized = _normalize_combo_param_names([name])
    return normalized[0] if normalized else None


def _parse_param_fraction_map(value, default: float = 1.0) -> Dict[str, float]:
    def _fill_defaults(overrides: Dict[str, float]) -> Dict[str, float]:
        filled = {name: float(default) for name in MOD_TARGET_PARAM_NAMES}
        for key, val in overrides.items():
            name = _normalize_param_name(key)
            if name is None:
                continue
            try:
                filled[name] = float(val)
            except (TypeError, ValueError):
                continue
        return filled

    if value is None:
        return {name: float(default) for name in MOD_TARGET_PARAM_NAMES}
    if isinstance(value, dict):
        return _fill_defaults(value)
    if isinstance(value, (list, tuple)):
        vals = list(value)
        if not vals:
            return {name: float(default) for name in MOD_TARGET_PARAM_NAMES}
        if len(vals) == 1:
            return {name: float(vals[0]) for name in MOD_TARGET_PARAM_NAMES}
        if len(vals) == 2:
            hidden_frac = float(vals[0])
            output_frac = float(vals[1])
            mapped = {name: hidden_frac for name in HIDDEN_PARAM_NAMES}
            mapped.update({name: output_frac for name in OUTPUT_PARAM_NAMES})
            return mapped
        mapped = {}
        for idx, name in enumerate(MOD_TARGET_PARAM_NAMES):
            if idx < len(vals):
                mapped[name] = float(vals[idx])
            else:
                mapped[name] = float(default)
        return mapped
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {name: float(default) for name in MOD_TARGET_PARAM_NAMES}
        parsed = None
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            try:
                parsed = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                parsed = None
        if parsed is not None:
            return _parse_param_fraction_map(parsed, default=default)
        parsed_list = parse_float_list(text)
        if parsed_list is not None:
            return _parse_param_fraction_map(parsed_list, default=default)
        return {name: float(default) for name in MOD_TARGET_PARAM_NAMES}
    try:
        scalar = float(value)
        return {name: scalar for name in MOD_TARGET_PARAM_NAMES}
    except (TypeError, ValueError):
        return {name: float(default) for name in MOD_TARGET_PARAM_NAMES}

def _resolve_ann_combo_lists(
    additive: Optional[List[str]],
    multiplicative: Optional[List[str]],
) -> Tuple[List[str], List[str]]:
    full = list(MOD_TARGET_PARAM_NAMES)
    add_set = set(_normalize_combo_param_names(additive))
    mult_set = set(_normalize_combo_param_names(multiplicative))
    if not add_set and not mult_set:
        add_set = {"thr", "reset", "rest"}
    if not add_set:
        add_set = set(full) - mult_set
    if not mult_set:
        mult_set = set(full) - add_set
    mult_set -= add_set
    missing = set(full) - add_set - mult_set
    mult_set |= missing
    add_list = [p for p in full if p in add_set]
    mult_list = [p for p in full if p in mult_set]
    return add_list, mult_list


def _parse_pair(arg, caster):
    if arg is None:
        return None
    values = arg
    if isinstance(arg, str):
        text = arg.strip()
        if not text:
            return None
        parsed = None
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            try:
                parsed = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                parsed = None
        if isinstance(parsed, (list, tuple)):
            return _parse_pair(parsed, caster)
        if parsed is not None:
            values = [parsed]
        else:
            values = [arg]
    elif isinstance(arg, (list, tuple)):
        vals = list(arg)
        if not vals:
            return None
        if len(vals) == 1:
            return _parse_pair(vals[0], caster)
        values = vals
    else:
        values = [values]
    if not values:
        return None
    if len(values) == 1:
        values = [values[0], values[0]]
    else:
        values = [values[0], values[1]]
    return (caster(values[0]), caster(values[1]))


def _cast_str(value):
    return str(value)


def _parse_int_pair(arg):
    return _parse_pair(arg, int)


def _parse_float_pair(arg):
    return _parse_pair(arg, float)


def _parse_str_pair(arg):
    pair = _parse_pair(arg, _cast_str)
    if not pair:
        return None
    return tuple(p.strip() for p in pair)

def _count_snn_params(state: Optional[Dict[str, torch.nn.Parameter]]) -> Tuple[int, int]:
    trainable = frozen = 0
    if not state:
        return trainable, frozen
    for tensor in state.values():
        if tensor is None or not hasattr(tensor, "numel"):
            continue
        num = int(tensor.numel())
        if isinstance(tensor, torch.nn.Parameter) and tensor.requires_grad:
            trainable += num
        else:
            frozen += num
    return trainable, frozen

def _count_module_params(module: Optional[nn.Module]) -> Tuple[int, int]:
    trainable = frozen = 0
    if module is None:
        return trainable, frozen
    for p in module.parameters():
        num = int(p.numel())
        if p.requires_grad:
            trainable += num
        else:
            frozen += num
    return trainable, frozen


def _clamp_state_params(state: Dict[str, torch.Tensor], ranges: Dict[str, Tuple[float, float]]):
    for tensor_name, range_name in PARAM_RANGE_TARGETS.items():
        tensor = state.get(tensor_name)
        if tensor is None:
            continue
        _clamp_param_tensor(tensor, range_name, ranges)


def _resolve_param_ranges(settings: Dict) -> Dict[str, Tuple[float, float]]:
    cached = settings.get("_param_ranges")
    if cached is not None:
        return cached
    ranges = {}
    user_ranges = settings.get("param_ranges", {})
    for name, default in PARAMETER_RANGE_DEFAULTS.items():
        rng = user_ranges.get(name)
        if rng is None:
            rng = settings.get(f"{name}_range")
        if rng is None:
            ranges[name] = tuple(default)
            continue
        if isinstance(rng, (list, tuple)) and len(rng) == 2:
            lo, hi = float(rng[0]), float(rng[1])
            if hi < lo:
                lo, hi = hi, lo
            ranges[name] = (lo, hi)
        else:
            ranges[name] = tuple(default)
    settings["_param_ranges"] = ranges
    return ranges


def _clamp_param_tensor(tensor: torch.Tensor, name: str, ranges: Dict[str, Tuple[float, float]]):
    lo, hi = ranges.get(name, PARAMETER_RANGE_DEFAULTS.get(name, (-float("inf"), float("inf"))))
    tensor.clamp_(float(lo), float(hi))


def _apply_additive_delta(base: torch.Tensor, delta: torch.Tensor, name: str,
                          ranges: Dict[str, Tuple[float, float]]) -> torch.Tensor:
    lo, hi = ranges.get(name, PARAMETER_RANGE_DEFAULTS.get(name, (-float("inf"), float("inf"))))
    return torch.clamp(base + delta, min=float(lo), max=float(hi))


def _map_substitution_output(raw: torch.Tensor, name: str, activation_kind: str,
                             ranges: Dict[str, Tuple[float, float]]) -> torch.Tensor:
    lo, hi = ranges.get(name, PARAMETER_RANGE_DEFAULTS.get(name, (-float("inf"), float("inf"))))
    act_entry = ANN_OUTPUT_ACTIVATIONS.get(activation_kind, (None, None))
    act_range = act_entry[1]

    # Convert MLP output into [0,1] regardless of the activation’s native range.
    if activation_kind == "linear":
        # For linear heads, bias toward midpoint: map 0 -> 0.5 and allow spread around it.
        norm = torch.clamp(raw + 0.5, 0.0, 1.0)
    elif act_range is None:
        norm = torch.sigmoid(raw)  # other unbounded -> squash
    else:
        act_lo, act_hi = act_range
        norm = (torch.clamp(raw, act_lo, act_hi) - act_lo) / max(act_hi - act_lo, 1e-6)
    norm = torch.clamp(norm, 0.0, 1.0)

    mapped = lo + norm * (hi - lo)
    return torch.clamp(mapped, min=float(lo), max=float(hi))


def _psp_peak_gain(alpha: torch.Tensor, beta: torch.Tensor, dt: float) -> torch.Tensor:
    """
    Analytic peak-normalizing gain for bi-exponential PSP.
    alpha = exp(-dt/tau_syn), beta = exp(-dt/tau_mem).
    gain = (tau_mem / tau_syn) ** (tau_mem / (tau_mem - tau_syn))
    Limit as tau_mem -> tau_syn is e.
    """
    eps = 1e-12
    # Recover continuous taus from decay factors and dt.
    tau_syn = -float(dt) / torch.log(torch.clamp(alpha, min=eps))
    tau_mem = -float(dt) / torch.log(torch.clamp(beta, min=eps))
    diff = tau_mem - tau_syn
    ratio = tau_mem / torch.clamp(tau_syn, min=eps)
    # Handle tau_mem ≈ tau_syn with the limit gain -> e.
    close = torch.isclose(diff, torch.zeros_like(diff), atol=1e-8, rtol=1e-6)
    gain = torch.exp(torch.log(torch.clamp(ratio, min=eps)) * (tau_mem / torch.clamp(diff, min=eps)))
    gain = torch.where(close, torch.full_like(gain, math.e), gain)
    gain = torch.clamp(gain, min=0.0)
    gain = torch.where(torch.isfinite(gain), gain, torch.ones_like(gain))
    return gain


def _init_input_delay_logits(settings: Dict, nb_inputs: int) -> torch.nn.Parameter:
    max_delay = max(1, int(settings.get("input_delay_steps", 1)))
    init_val = settings.get("input_delay_init", None)
    init_cap = settings.get("input_delay_init_cap", max_delay)
    if init_cap is None:
        init_cap = max_delay
    init_cap = int(init_cap)
    init_cap = max(0, min(init_cap, max_delay))
    logits = torch.zeros((nb_inputs, max_delay + 1), device=device, dtype=dtype, requires_grad=True)
    noise_scale = float(settings.get("input_delay_init_noise", 0.01))
    if noise_scale > 0:
        logits.data += noise_scale * torch.randn_like(logits)
    with torch.no_grad():
        if init_val is None:
            # Optional small bias toward random delays in [0, init_cap] without saturating logits
            if init_cap >= 0:
                init_delays = torch.randint(0, init_cap + 1, (nb_inputs,), device=device)
                logits[torch.arange(nb_inputs, device=device), init_delays] += float(settings.get("input_delay_init_bias", 0.1))
        else:
            init_delays = torch.full((nb_inputs,), min(max_delay, max(0, int(init_val))), device=device, dtype=torch.long)
            logits[torch.arange(nb_inputs, device=device), init_delays] += float(settings.get("input_delay_init_bias", 0.1))
    return logits


def _next_power_of_two(n: int) -> int:
    n = max(1, int(n))
    pow2 = 1
    while pow2 < n:
        pow2 <<= 1
    return pow2




# -------------------------
# Tau/time-step helpers
# -------------------------
def _effective_tau_decay_dt(settings: Dict[str, Union[int, float, bool]]) -> float:
    dt = float(settings.get("time_step", 1.0))
    if settings.get("tau_match_clip") and settings.get("max_time", 0.0) > 0:
        steps = max(1, int(settings.get("nb_steps", 1)))
        return float(settings["max_time"]) / steps
    return dt




__all__ = [name for name in globals() if not name.startswith('__')]
