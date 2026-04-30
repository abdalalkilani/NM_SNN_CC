# =============================================================================
# SNN ALL-IN-ONE SMALL
#
# Kept: base SNN, data loading/splits, channel compression, ANN add/sub MLP,
#       SNN additive modulator, modulated/staged training, neuromodulators,
#       grouping, fixed masks, param controls, model helpers.
# Removed: input delay, ANN combo, RNN/LSTM modulators, SNN substitution, NM debug print.
# =============================================================================

# =============================================================================
# SECTION 1: Imports, Device, Constants
# =============================================================================

import os, math, random, ast, copy, json
from pathlib import Path
from typing import Any, Dict, Tuple, Union, Optional, List, Callable

import h5py, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F

# Device / dtype
dtype = torch.float
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

SNN_PARAM_NAMES = [
    "w1", "w2", "v1",
    "alpha_hetero_1", "beta_hetero_1",
    "alpha_hetero_2", "beta_hetero_2",
    "thresholds_1", "reset_1", "rest_1",
]
MOD_TARGET_PARAM_NAMES = ["alpha_1", "beta_1", "thr", "reset", "rest", "alpha_2", "beta_2"]
HIDDEN_PARAM_NAMES = ["alpha_1", "beta_1", "thr", "reset", "rest"]
OUTPUT_PARAM_NAMES = ["alpha_2", "beta_2"]

PARAMETER_RANGE_DEFAULTS = {
    "alpha_1": (1.0 / math.e, 0.995),
    "beta_1": (1.0 / math.e, 0.995),
    "alpha_2": (1.0 / math.e, 0.995),
    "beta_2": (1.0 / math.e, 0.995),
    "thr": (0.5, 1.5),
    "reset": (-0.5, 0.5),
    "rest": (-0.5, 0.5),
}
PARAM_RANGE_TARGETS = {
    "alpha_hetero_1": "alpha_1",
    "beta_hetero_1": "beta_1",
    "alpha_hetero_2": "alpha_2",
    "beta_hetero_2": "beta_2",
    "thresholds_1": "thr",
    "reset_1": "reset",
    "rest_1": "rest",
}
ANN_OUTPUT_ACTIVATIONS = {
    "sigmoid": (nn.Sigmoid, (0.0, 1.0)),
    "tanh": (nn.Tanh, (-1.0, 1.0)),
    "linear": (nn.Identity, None),
}
GROUP_DISTRIBUTIONS = {"uniform", "normal"}
GROUP_NORMAL_CUTOFF = 0.05
PARAM_SMOOTH_TAU_INIT_DEFAULT = 0.2  # per-step mixing fraction (0-1)
PARAM_SMOOTH_TAU_MIN_DEFAULT = 0.0


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

# =============================================================================
# SECTION 2: General Utilities and Parameter Helpers
# =============================================================================
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



def _next_power_of_two(n: int) -> int:
    n = max(1, int(n))
    pow2 = 1
    while pow2 < n:
        pow2 <<= 1
    return pow2


# =============================================================================
# SECTION 3: Neuromodulator Mapping
# =============================================================================

class NeuromodulatorMapper(nn.Module):
    """Shared per-layer MLPs that map neuromodulator levels to parameter deltas/values."""

    def __init__(
        self,
        hidden_targets: int,
        output_targets: int,
        hidden_per_neuron: int,
        output_per_neuron: int,
        init_scale: float,
        cfg: Optional[Dict] = None,
    ):
        super().__init__()
        cfg = cfg or {}
        self.hidden_targets = max(0, int(hidden_targets))
        self.output_targets = max(0, int(output_targets))
        self.hidden_per_neuron = max(0, int(hidden_per_neuron))
        self.output_per_neuron = max(0, int(output_per_neuron))
        self.init_scale = float(init_scale)
        self.activation_kind = cfg.get("activation_kind")
        self.mapper_type = str(cfg.get("mapper_type", "mlp")).lower()
        self.hidden_activation = str(cfg.get("hidden_activation", "silu") or "silu").lower()
        self.flat_order = str(cfg.get("flat_order", "type_major") or "type_major").lower()
        if self.flat_order not in {"type_major", "target_major"}:
            self.flat_order = "type_major"
        self.init_identity = bool(cfg.get("init_identity", False))
        self.hidden_width = self._resolve_width(self.hidden_per_neuron, 5, cfg, "hidden_mlp_hidden")
        self.output_width = self._resolve_width(self.output_per_neuron, 2, cfg, "output_mlp_hidden")

        self.hidden_mlp: Optional[nn.Module] = None
        self.output_mlp: Optional[nn.Module] = None
        if self.hidden_per_neuron > 0 and self.hidden_targets > 0:
            self.hidden_mlp = self._build_mlp(self.hidden_per_neuron, self.hidden_width, 5)
        if self.output_per_neuron > 0 and self.output_targets > 0:
            self.output_mlp = self._build_mlp(self.output_per_neuron, self.output_width, 2)

    @staticmethod
    def _resolve_width(in_dim: int, out_dim: int, cfg: Optional[Dict], cfg_key: str) -> int:
        if cfg:
            override = cfg.get(cfg_key, None)
            if override is None:
                override = cfg.get("mlp_hidden", None)
            if override is not None:
                if cfg.get("mapper_hidden_exact"):
                    return max(1, int(override))
                # Ensure at least 4x the larger of in/out even when overridden.
                return max(1, 20 * max(in_dim, out_dim), int(override))
        base = max(in_dim, out_dim)
        return max(1, 20 * base)

    def _build_mlp(self, in_dim: int, hidden_dim: int, out_dim: int) -> nn.Module:
        def _final_activation(kind: Optional[str]) -> nn.Module:
            k = (kind or "").lower()
            if k == "sigmoid":
                return nn.Sigmoid()
            if k == "tanh":
                return nn.Tanh()
            return nn.Identity()
        def _hidden_activation(kind: str) -> nn.Module:
            k = (kind or "silu").lower()
            if k == "relu":
                return nn.ReLU()
            if k == "gelu":
                return nn.GELU()
            if k == "leakyrelu":
                return nn.LeakyReLU(0.1)
            if k == "tanh":
                return nn.Tanh()
            if k == "none":
                return nn.Identity()
            return nn.SiLU()
        final_act = _final_activation(self.activation_kind)
        if self.mapper_type == "linear":
            mlp = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                final_act,
            )
            linear_layers = [m for m in mlp.modules() if isinstance(m, nn.Linear)]
        else:
            hidden_act = _hidden_activation(self.hidden_activation)
            mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                hidden_act,
                nn.Linear(hidden_dim, hidden_dim),
                hidden_act,
                nn.Linear(hidden_dim, out_dim),
                final_act,
            )
            linear_layers = [m for m in mlp.modules() if isinstance(m, nn.Linear)]
        # Init: default small normal for all layers; final layer bias set for neutral outputs.
        for idx, layer in enumerate(linear_layers):
            nn.init.normal_(layer.weight, mean=0.0, std=self.init_scale)
            bias_val = 0.0
            if idx == len(linear_layers) - 1 and (self.activation_kind or "").lower() == "sigmoid":
                # sigmoid(0)=0.5 midpoint
                bias_val = 0.0
            nn.init.constant_(layer.bias, bias_val)
        # Optional warm-start: identity mapping (logit passthrough) for linear mappers.
        if self.init_identity and linear_layers:
            # For MLP mappers, only apply identity init when hidden activation is linear (none),
            # otherwise the nonlinearity destroys the warm-start behavior.
            if self.mapper_type == "linear":
                lin = linear_layers[0]
                with torch.no_grad():
                    lin.weight.zero_()
                    lin.bias.zero_()
                    diag = min(out_dim, in_dim)
                    if diag > 0:
                        eye = torch.eye(diag, dtype=lin.weight.dtype, device=lin.weight.device)
                        lin.weight[:diag, :diag].copy_(eye)
            elif self.mapper_type == "mlp" and self.hidden_activation == "none" and len(linear_layers) >= 3:
                l0, l1, l2 = linear_layers[0], linear_layers[1], linear_layers[2]
                with torch.no_grad():
                    for lin in (l0, l1, l2):
                        lin.weight.zero_()
                        lin.bias.zero_()
                    # l0: embed input into first dims of hidden.
                    d0 = min(l0.out_features, l0.in_features)
                    if d0 > 0:
                        eye0 = torch.eye(d0, dtype=l0.weight.dtype, device=l0.weight.device)
                        l0.weight[:d0, :d0].copy_(eye0)
                    # l1: hidden identity.
                    d1 = min(l1.out_features, l1.in_features)
                    if d1 > 0:
                        eye1 = torch.eye(d1, dtype=l1.weight.dtype, device=l1.weight.device)
                        l1.weight[:d1, :d1].copy_(eye1)
                    # l2: project first dims back to output.
                    d2 = min(l2.out_features, l2.in_features)
                    if d2 > 0:
                        eye2 = torch.eye(d2, dtype=l2.weight.dtype, device=l2.weight.device)
                        l2.weight[:d2, :d2].copy_(eye2)
        return mlp

    @property
    def hidden_flat_dim(self) -> int:
        return self.hidden_targets * self.hidden_per_neuron

    @property
    def output_flat_dim(self) -> int:
        return self.output_targets * self.output_per_neuron

    @property
    def total_output_dim(self) -> int:
        return self.hidden_flat_dim + self.output_flat_dim

    def _split_levels(self, flat: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        offset = 0
        hidden_levels = output_levels = None
        if self.hidden_flat_dim > 0:
            hidden_slice = flat[:, offset:offset + self.hidden_flat_dim]
            if self.flat_order == "type_major":
                hidden_levels = hidden_slice.view(flat.size(0), self.hidden_per_neuron, self.hidden_targets).permute(0, 2, 1)
            else:
                hidden_levels = hidden_slice.view(flat.size(0), self.hidden_targets, self.hidden_per_neuron)
            offset += self.hidden_flat_dim
        if self.output_flat_dim > 0:
            out_slice = flat[:, offset:offset + self.output_flat_dim]
            if self.flat_order == "type_major":
                output_levels = out_slice.view(flat.size(0), self.output_per_neuron, self.output_targets).permute(0, 2, 1)
            else:
                output_levels = out_slice.view(flat.size(0), self.output_targets, self.output_per_neuron)
        return hidden_levels, output_levels

    def effects_from_levels(
        self,
        hidden_levels: Optional[torch.Tensor],
        output_levels: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        ref = hidden_levels if hidden_levels is not None else output_levels
        batch = ref.size(0) if ref is not None else 0
        device = ref.device if ref is not None else None
        dtype = ref.dtype if ref is not None else None
        effects = {
            "alpha_1": torch.zeros((batch, self.hidden_targets), device=device, dtype=dtype),
            "beta_1": torch.zeros((batch, self.hidden_targets), device=device, dtype=dtype),
            "thr": torch.zeros((batch, self.hidden_targets), device=device, dtype=dtype),
            "reset": torch.zeros((batch, self.hidden_targets), device=device, dtype=dtype),
            "rest": torch.zeros((batch, self.hidden_targets), device=device, dtype=dtype),
            "alpha_2": torch.zeros((batch, self.output_targets), device=device, dtype=dtype),
            "beta_2": torch.zeros((batch, self.output_targets), device=device, dtype=dtype),
        }
        if hidden_levels is not None and self.hidden_mlp is not None:
            h = self.hidden_mlp(hidden_levels.reshape(-1, self.hidden_per_neuron))
            h = h.reshape(hidden_levels.size(0), self.hidden_targets, 5)
            effects["alpha_1"] = h[:, :, 0]
            effects["beta_1"] = h[:, :, 1]
            effects["thr"] = h[:, :, 2]
            effects["reset"] = h[:, :, 3]
            effects["rest"] = h[:, :, 4]
        if output_levels is not None and self.output_mlp is not None:
            o = self.output_mlp(output_levels.reshape(-1, self.output_per_neuron))
            o = o.reshape(output_levels.size(0), self.output_targets, 2)
            effects["alpha_2"] = o[:, :, 0]
            effects["beta_2"] = o[:, :, 1]
        return effects

    def effects_from_flat(self, flat: torch.Tensor) -> Dict[str, torch.Tensor]:
        hidden_levels, output_levels = self._split_levels(flat)
        return self.effects_from_levels(hidden_levels, output_levels)


# =============================================================================
# SECTION 4: Grouping, Overlap, and Fixed Modulation Masks
# =============================================================================

class GroupLayout:
    def __init__(self, target_count: int, size: int, overlap: int, distribution: str, normal_std: float):
        self.target_count = max(1, int(target_count))
        self.size = max(1, min(int(size), self.target_count))
        self.overlap = max(0, min(int(overlap), self.size - 1))
        self.distribution = distribution.lower() if distribution else "uniform"
        if self.distribution not in GROUP_DISTRIBUTIONS:
            self.distribution = "uniform"
        self.normal_std = float(normal_std if normal_std is not None else 1.0)
        self.enabled = (self.size > 1) or (self.overlap > 0)
        self._expand_divisor = None
        if not self.enabled:
            self.group_count = self.target_count
            self.forward = None
            self.backward = None
            return
        is_normal = self.distribution == "normal"
        stride = max(1, self.size if is_normal else (self.size - self.overlap))
        groups = max(1, math.ceil(self.target_count / stride))
        self.group_count = groups
        forward = np.zeros((groups, self.target_count), dtype=np.float32)
        for g in range(groups):
            if self.distribution == "uniform":
                start = (g * stride) % self.target_count
                for k in range(self.size):
                    idx = (start + k) % self.target_count
                    forward[g, idx] += 1.0
                continue

            # Normal: soft Gaussian falloff over all targets. Overlap is redundant here; spacing uses size.
            center = (g * stride + (self.size - 1) / 2.0) % self.target_count
            denom = max(self.normal_std, 1e-6)
            for idx in range(self.target_count):
                dist = abs(idx - center)
                if dist > self.target_count / 2:
                    dist = self.target_count - dist
                weight = math.exp(-0.5 * (dist / denom) ** 2)
                if weight >= GROUP_NORMAL_CUTOFF:
                    forward[g, idx] += weight
        forward_tensor = torch.from_numpy(forward)
        if self.distribution == "uniform":
            self._expand_divisor = forward_tensor.sum(dim=0, keepdim=True).clamp_min(1e-6)
        norm = forward_tensor.sum(dim=1, keepdim=True).clamp_min(1e-6)
        backward = forward_tensor / norm
        self.forward = forward_tensor
        self.backward = backward

    def project(self, values: torch.Tensor) -> torch.Tensor:
        if not self.enabled or self.forward is None:
            return values
        weights = self.forward.t().to(device=values.device, dtype=values.dtype)
        return torch.matmul(values, weights)

    def enforce(self, values: torch.Tensor) -> torch.Tensor:
        if not self.enabled or self.forward is None or self.backward is None:
            return values
        back = self.backward.to(device=values.device, dtype=values.dtype)
        fwd = self.forward.to(device=values.device, dtype=values.dtype)
        group_vals = torch.matmul(values, back.t())
        return torch.matmul(group_vals, fwd)

    def expand(self, grouped: torch.Tensor) -> torch.Tensor:
        if not self.enabled or self.forward is None:
            return grouped
        weights = self.forward.to(device=grouped.device, dtype=grouped.dtype)
        expanded = torch.matmul(grouped, weights)
        if self._expand_divisor is not None and self.distribution == "uniform":
            divisor = self._expand_divisor.to(device=expanded.device, dtype=expanded.dtype)
            expanded = expanded / divisor
        return expanded


def _uniform_group_layout(layout: Optional[GroupLayout]) -> Optional[GroupLayout]:
    if layout is None or not layout.enabled:
        return layout
    if layout.distribution == "uniform":
        return layout
    return GroupLayout(layout.target_count, layout.size, layout.overlap, "uniform", layout.normal_std)


def _group_layer_cfg(settings: Dict, layer_index: int) -> Dict:
    base = settings.get("group_cfg")
    if not base:
        return {}
    def _select(key, default):
        pair = base.get(key)
        if isinstance(pair, (list, tuple)):
            if len(pair) > layer_index:
                return pair[layer_index]
            return pair[0]
        return pair if pair is not None else default
    size = max(1, int(_select("size", 1)))
    overlap = max(0, int(_select("overlap", 0)))
    enable_override = _select("enable", None)
    enabled = bool(enable_override) if enable_override is not None else (size > 1 or overlap > 0)
    return {
        "enable": enabled,
        "size": size,
        "overlap": overlap,
        "distribution": str(_select("distribution", "uniform")),
        "normal_std": float(_select("normal_std", 1.0)),
    }


def _get_group_layout(settings: Dict, layer_index: int, target_count: int) -> Optional[GroupLayout]:
    cfg = _group_layer_cfg(settings, layer_index)
    if not cfg.get("enable"):
        return None
    cache = settings.setdefault("_group_layout_cache", {})
    key = (layer_index, target_count, cfg["size"], cfg["overlap"], cfg["distribution"], cfg["normal_std"])
    layout = cache.get(key)
    if layout is None:
        layout = GroupLayout(target_count, cfg["size"], cfg["overlap"], cfg["distribution"], cfg["normal_std"])
        cache[key] = layout
    return layout


def _ann_out_size_overrides(settings: Dict) -> Dict[str, int]:
    overrides: Dict[str, int] = {}
    nb_hidden = settings["nb_hidden"]
    nb_outputs = settings["nb_outputs"]
    hidden_layout = _get_group_layout(settings, 0, nb_hidden)
    output_layout = _get_group_layout(settings, 1, nb_outputs)
    hidden_groups = hidden_layout.group_count if hidden_layout else nb_hidden
    out_groups = output_layout.group_count if output_layout else nb_outputs
    for name in ("alpha_1", "beta_1", "thr", "reset", "rest"):
        overrides[name] = hidden_groups
    overrides["alpha_2"] = out_groups
    overrides["beta_2"] = out_groups
    return overrides


def _ann_in_size_overrides(settings: Dict, substitution_mode: bool) -> Dict[str, int]:
    if not substitution_mode:
        return {}
    overrides: Dict[str, int] = {}
    nb_hidden = settings["nb_hidden"]
    nb_outputs = settings["nb_outputs"]
    hidden_layout = _get_group_layout(settings, 0, nb_hidden)
    output_layout = _get_group_layout(settings, 1, nb_outputs)
    hidden_groups = hidden_layout.group_count if hidden_layout else nb_hidden
    out_groups = output_layout.group_count if output_layout else nb_outputs
    for name in ("alpha_1", "beta_1", "thr", "reset", "rest"):
        overrides[name] = hidden_groups
    overrides["alpha_2"] = out_groups
    overrides["beta_2"] = out_groups
    return overrides


def _select_fraction_indices(count: int, frac: float, generator: Optional[torch.Generator], ensure_min: bool = True) -> torch.Tensor:
    count = max(0, int(count))
    if count == 0:
        return torch.empty((0,), dtype=torch.long)
    frac = float(frac)
    if frac >= 1.0:
        idx = torch.arange(count, dtype=torch.long)
    elif frac <= 0.0:
        idx = torch.empty((0,), dtype=torch.long)
    else:
        mask = torch.rand(count, generator=generator) < frac
        idx = mask.nonzero(as_tuple=False).flatten().to(dtype=torch.long)
    if ensure_min and idx.numel() == 0 and frac > 0.0:
        if generator is None:
            pick = torch.randint(0, count, (1,))
        else:
            pick = torch.randint(0, count, (1,), generator=generator)
        idx = pick.to(dtype=torch.long)
    return idx


def _build_fixed_mod_mask(
    settings: Dict,
    hidden_layout: Optional['GroupLayout'],
    output_layout: Optional['GroupLayout']
) -> Optional[Dict[str, torch.Tensor]]:
    cfg = settings.get("mod_mask_cfg", {}) or {}
    zero_fallback = bool(cfg.get("zero_fallback", False))
    if not cfg.get("fixed_enable"):
        return None
    nm_cfg = settings.get("nm_cfg", {}) or {}
    nm_neuron_frac_enable = bool(nm_cfg.get("neuron_fraction_enable", False))
    nm_param_frac_enable = bool(nm_cfg.get("param_fraction_enable", False))
    if not (nm_neuron_frac_enable or nm_param_frac_enable):
        return None
    cached = settings.get("_mod_fixed_mask")
    if cached is not None:
        return cached

    def _pair(val, default):
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

    nm_neuron_frac = _pair(nm_cfg.get("neuron_fraction", (1.0, 1.0)), (1.0, 1.0))
    nm_param_frac_map = _parse_param_fraction_map(nm_cfg.get("param_fraction", None), default=1.0)
    nb_hidden = int(settings.get("nb_hidden", 0))
    nb_outputs = int(settings.get("nb_outputs", 0))
    hidden_target = hidden_layout.group_count if hidden_layout else nb_hidden
    output_target = output_layout.group_count if output_layout else nb_outputs
    hidden_param_fracs = [float(nm_param_frac_map.get(name, 1.0)) for name in HIDDEN_PARAM_NAMES]
    output_param_fracs = [float(nm_param_frac_map.get(name, 1.0)) for name in OUTPUT_PARAM_NAMES]
    allow_empty_hidden = (nm_neuron_frac_enable and float(nm_neuron_frac[0]) <= 0.0) or (
        nm_param_frac_enable and max(hidden_param_fracs) <= 0.0
    )
    allow_empty_output = (nm_neuron_frac_enable and float(nm_neuron_frac[1]) <= 0.0) or (
        nm_param_frac_enable and max(output_param_fracs) <= 0.0
    )
    if zero_fallback:
        allow_empty_hidden = False
        allow_empty_output = False

    gen = None
    seed = cfg.get("fixed_seed")
    if seed is None:
        seed = cfg.get("seed")
    if seed is not None:
        gen = torch.Generator()
        gen.manual_seed(int(seed))

    if nm_neuron_frac_enable:
        hidden_neuron_idx = _select_fraction_indices(hidden_target, nm_neuron_frac[0], gen)
        output_neuron_idx = _select_fraction_indices(output_target, nm_neuron_frac[1], gen)
    else:
        hidden_neuron_idx = torch.arange(hidden_target, dtype=torch.long)
        output_neuron_idx = torch.arange(output_target, dtype=torch.long)

    def _select_param_mask_indices(base_idx: torch.Tensor, frac: float) -> torch.Tensor:
        if not nm_param_frac_enable:
            return base_idx
        if base_idx.numel() == 0:
            return base_idx
        mask = torch.rand(base_idx.numel(), generator=gen) < float(frac)
        idx = base_idx[mask].to(dtype=torch.long)
        if idx.numel() == 0 and frac > 0.0:
            if gen is None:
                pick = torch.randint(0, base_idx.numel(), (1,))
            else:
                pick = torch.randint(0, base_idx.numel(), (1,), generator=gen)
            idx = base_idx[pick].to(dtype=torch.long)
        return idx

    hidden_param_mask_idx = {
        name: _select_param_mask_indices(hidden_neuron_idx, nm_param_frac_map.get(name, 1.0))
        for name in HIDDEN_PARAM_NAMES
    }
    output_param_mask_idx = {
        name: _select_param_mask_indices(output_neuron_idx, nm_param_frac_map.get(name, 1.0))
        for name in OUTPUT_PARAM_NAMES
    }

    def _ensure_io_indices(
        mask_idx: torch.Tensor,
        base_idx: torch.Tensor,
        target_count: int,
        allow_empty: bool = False
    ) -> torch.Tensor:
        if mask_idx.numel() > 0:
            return mask_idx
        if allow_empty:
            return mask_idx
        if base_idx.numel() > 0:
            return base_idx
        if target_count > 0:
            return torch.arange(target_count, dtype=torch.long)
        return torch.empty((0,), dtype=torch.long)

    hidden_param_idx = {
        name: _ensure_io_indices(
            hidden_param_mask_idx[name],
            hidden_neuron_idx,
            hidden_target,
            allow_empty=allow_empty_hidden,
        )
        for name in HIDDEN_PARAM_NAMES
    }
    output_param_idx = {
        name: _ensure_io_indices(
            output_param_mask_idx[name],
            output_neuron_idx,
            output_target,
            allow_empty=allow_empty_output,
        )
        for name in OUTPUT_PARAM_NAMES
    }

    def _union_indices(indices: List[torch.Tensor], target_count: int, allow_empty: bool) -> torch.Tensor:
        if not indices or target_count <= 0:
            return torch.empty((0,), dtype=torch.long)
        non_empty = [idx for idx in indices if idx.numel() > 0]
        if not non_empty:
            if allow_empty:
                return torch.empty((0,), dtype=torch.long)
            return _select_fraction_indices(target_count, 1.0, gen)
        merged = torch.cat(non_empty, dim=0)
        if merged.numel() == 0:
            return torch.empty((0,), dtype=torch.long)
        merged = torch.unique(merged)
        merged, _ = torch.sort(merged)
        if merged.numel() == 0 and target_count > 0:
            if allow_empty:
                return torch.empty((0,), dtype=torch.long)
            merged = _select_fraction_indices(target_count, 1.0, gen)
        return merged.to(dtype=torch.long)

    hidden_union_idx = _union_indices(list(hidden_param_idx.values()), hidden_target, allow_empty_hidden)
    output_union_idx = _union_indices(list(output_param_idx.values()), output_target, allow_empty_output)

    def _indices_to_mask(count: int, idx: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros((max(0, int(count)),), dtype=torch.float32)
        if idx.numel() > 0:
            mask[idx] = 1.0
        return mask

    param_masks_target: Dict[str, torch.Tensor] = {}
    for name, idx in {**hidden_param_mask_idx, **output_param_mask_idx}.items():
        count = hidden_target if name in HIDDEN_PARAM_NAMES else output_target
        param_masks_target[name] = _indices_to_mask(count, idx)

    def _mask_in_union(union_idx: torch.Tensor, group_idx: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros((union_idx.numel(),), dtype=torch.float32)
        if union_idx.numel() == 0 or group_idx.numel() == 0:
            return mask
        positions = {int(v): i for i, v in enumerate(union_idx.tolist())}
        for v in group_idx.tolist():
            pos = positions.get(int(v))
            if pos is not None:
                mask[pos] = 1.0
        return mask

    param_masks_union: Dict[str, torch.Tensor] = {}
    for name, idx in hidden_param_mask_idx.items():
        param_masks_union[name] = _mask_in_union(hidden_union_idx, idx)
    for name, idx in output_param_mask_idx.items():
        param_masks_union[name] = _mask_in_union(output_union_idx, idx)

    flat_inputs = bool(cfg.get("fixed_flat_inputs", False))
    flat_hidden_idx = None
    flat_output_idx = None
    if flat_inputs and nm_neuron_frac_enable:
        flat_hidden_idx = _select_fraction_indices(nb_hidden, nm_neuron_frac[0], gen)
        flat_output_idx = _select_fraction_indices(nb_outputs, nm_neuron_frac[1], gen)

    mask_cfg = {
        "hidden_target": int(hidden_target),
        "output_target": int(output_target),
        "hidden_neuron_idx": hidden_neuron_idx,
        "output_neuron_idx": output_neuron_idx,
        "hidden_param_idx": hidden_param_idx,
        "output_param_idx": output_param_idx,
        "hidden_union_idx": hidden_union_idx,
        "output_union_idx": output_union_idx,
        "param_masks_target": param_masks_target,
        "param_masks_union": param_masks_union,
        "flat_inputs": flat_inputs,
        "flat_hidden_idx": flat_hidden_idx,
        "flat_output_idx": flat_output_idx,
    }
    settings["_mod_fixed_mask"] = mask_cfg
    return mask_cfg


# =============================================================================
# SECTION 5: Parameter Timescales and Smoothing
# =============================================================================

class ParamTimescaleController(nn.Module):
    def __init__(
        self,
        base_interval: int,
        names: List[str],
        dist: str = "fixed",
        scale: float = 0.0,
        std: float = 0.0,
        seed: Optional[int] = None,
        trainable: bool = True,
    ):
        super().__init__()
        self.base_interval = max(1, int(base_interval))
        self.names = list(names)
        self.dist = dist.lower() if isinstance(dist, str) else "fixed"
        self.scale = max(0.0, float(scale))
        self.std = max(0.0, float(std))
        rng = np.random.default_rng(seed) if seed is not None else np.random
        init = np.zeros(len(self.names), dtype=np.float32)
        if self.dist == "uniform" and self.scale > 0:
            init = rng.uniform(0.0, self.scale, size=len(self.names))
        elif self.dist == "normal" and (self.std > 0 or self.scale > 0):
            stdev = self.std if self.std > 0 else self.scale
            init = rng.normal(loc=0.0, scale=stdev, size=len(self.names))
            init = np.clip(init, 0.0, None)
        self.offsets = nn.Parameter(torch.from_numpy(init).float(), requires_grad=trainable)

    def intervals(self) -> Dict[str, torch.Tensor]:
        deltas = F.softplus(self.offsets)
        base = torch.tensor(float(self.base_interval), device=deltas.device, dtype=deltas.dtype)
        intervals = base + deltas
        # Straight-through rounding: forward uses round, backward flows through the continuous value.
        ste = (torch.round(intervals) - intervals).detach() + intervals
        ste = torch.clamp(ste, min=base)
        return {name: ste[idx] for idx, name in enumerate(self.names)}


class ParamSmoothingController(nn.Module):
    """
    Learnable per-parameter mixing fractions (0-1) applied each timestep toward targets.
    """

    def __init__(
        self,
        names: List[str],
        hidden_size: int,
        output_size: int,
        tau_init: float = PARAM_SMOOTH_TAU_INIT_DEFAULT,
        tau_min: float = PARAM_SMOOTH_TAU_MIN_DEFAULT,
        trainable: bool = True,
    ):
        super().__init__()
        self.names = list(names)
        self.hidden_size = int(hidden_size)
        self.output_size = int(output_size)
        self.mix_min = 0.0
        init = float(tau_init)
        init = min(max(init, self.mix_min), 0.9999)

        def _logit_scalar(x: float) -> float:
            eps = 1e-4
            x = max(min(float(x), 1 - eps), eps)
            return math.log(x / (1.0 - x))

        params = {}
        for name in self.names:
            dim = self.hidden_size if name in ("alpha_1", "beta_1", "thr", "reset", "rest") else self.output_size
            params[name] = nn.Parameter(torch.full((1, dim), float(_logit_scalar(init))), requires_grad=trainable)
        self.raw_mix = nn.ParameterDict(params)

    def mixes(self) -> Dict[str, torch.Tensor]:
        mixes = {}
        for name, tensor in self.raw_mix.items():
            mix = torch.sigmoid(tensor)
            mix = torch.clamp(mix, min=0.0, max=1.0)
            mixes[name] = mix
        return mixes

    def mixing_factors(self) -> Dict[str, torch.Tensor]:
        return self.mixes()


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


# =============================================================================
# SECTION 6: Data Loading, Augmentation, Channel Compression, and Splits
# =============================================================================
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
# =============================================================================
# SECTION 7: Base SNN Model and Forward Pass
# =============================================================================

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
def _effective_tau_decay_dt(settings: Dict[str, Union[int, float, bool]]) -> float:
    dt = float(settings.get("time_step", 1.0))
    if settings.get("tau_match_clip") and settings.get("max_time", 0.0) > 0:
        steps = max(1, int(settings.get("nb_steps", 1)))
        return float(settings["max_time"]) / steps
    return dt


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


    # --------------------------------

    state = dict(
        thresholds_1=thresholds_1, reset_1=reset_1, rest_1=rest_1,
        alpha_hetero_1=alpha_hetero_1, beta_hetero_1=beta_hetero_1,
        alpha_hetero_2=alpha_hetero_2, beta_hetero_2=beta_hetero_2,
        w1=w1, w2=w2, v1=v1,
    )
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
    h1_from_input = torch.einsum("btc,cd->btd", inputs_dense, w1)

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

# =============================================================================
# SECTION 8: ANN/SNN Modulator Models
# =============================================================================
def _mlp_io_slices(nb_inputs, nb_hidden, nb_outputs, in_mask: Dict, out_mask: Dict,
                   in_size_override: Optional[Dict[str, int]] = None,
                   out_size_override: Optional[Dict[str, int]] = None):
    in_groups = [
        ("alpha_1", nb_hidden), ("beta_1", nb_hidden),
        ("thr", nb_hidden), ("reset", nb_hidden), ("rest", nb_hidden),
        ("alpha_2", nb_outputs), ("beta_2", nb_outputs),
        ("in_flat", nb_inputs), ("hid_flat", nb_hidden), ("out_flat", nb_outputs)
    ]
    out_groups = [
        ("alpha_1", nb_hidden), ("beta_1", nb_hidden),
        ("thr", nb_hidden), ("reset", nb_hidden), ("rest", nb_hidden),
        ("alpha_2", nb_outputs), ("beta_2", nb_outputs)
    ]
    in_slices, in_size, cur = {}, 0, 0
    for name, size in in_groups:
        if in_mask.get(name, True):
            override = (in_size_override or {}).get(name)
            length = int(override) if override is not None else size
            in_slices[name] = slice(cur, cur+length); cur += length
    in_size = cur
    out_slices, out_size, cur = {}, 0, 0
    for name, size in out_groups:
        if out_mask.get(name, True):
            override = (out_size_override or {}).get(name)
            length = int(override) if override is not None else size
            out_slices[name] = slice(cur, cur+length); cur += length
    out_size = cur
    return in_slices, out_slices, in_size, out_size

DEFAULT_MLP_MODE = "mlp_sub"
DEFAULT_MLP_ARCH = "mlp"
VALID_MLP_ARCHS = {"mlp"}
MLP_MODE_SPECS = {
    "mlp_sub": {"activation": nn.Sigmoid, "activation_name": "sigmoid", "kind": "mlp"},
    "mlp_add": {"activation": nn.Tanh, "activation_name": "tanh", "kind": "mlp"},
    "snn_add": {"activation": nn.Identity, "activation_name": "linear", "kind": "snn"},
}

ADDITIVE_CLAMP_RANGES = {
    "alpha_1": (1.0/math.e, 0.995),
    "beta_1":  (1.0/math.e, 0.995),
    "alpha_2": (1.0/math.e, 0.995),
    "beta_2":  (1.0/math.e, 0.995),
    "thr":     (0.5, 1.5),
    "reset":   (-0.5, 0.5),
    "rest":    (-0.5, 0.5),
}
TINY_MOD_INIT_SCALE = 1e-7
TINY_MOD_GAIN_INIT = 1e-7
SOFT_MOD_INIT_SCALE = 1e-4
SOFT_MOD_GAIN_INIT = 1e-4
SNN_ADD_WEIGHT_SCALE_DEFAULT = 1e-2
SNN_ADD_GAIN_INIT_DEFAULT = 1e-2
SNN_SUB_SCALE_INIT_DEFAULT = 1.0
SNN_SUB_BIAS_INIT_DEFAULT = 0.0
ANN_ADD_FINAL_STD = 1e-4
COMBO_MULT_SCALE_MIN = 1.0 / 3.0
COMBO_MULT_SCALE_MAX = 3.0
COMBO_MULT_LOG_SPAN = math.log(COMBO_MULT_SCALE_MAX)


def _init_tiny_diag(matrix: torch.Tensor, scale: float = TINY_MOD_INIT_SCALE):
    """Sparse, near-zero init: zero everywhere with a tiny diagonal."""
    if matrix is None or matrix.dim() < 2:
        return
    with torch.no_grad():
        matrix.zero_()
        if scale <= 0:
            return
        diag = min(matrix.size(0), matrix.size(1))
        if diag > 0:
            eye = torch.eye(diag, dtype=matrix.dtype, device=matrix.device) * float(scale)
            matrix[:diag, :diag] = eye


def _init_tiny_linear(lin: nn.Linear, scale: float = TINY_MOD_INIT_SCALE):
    if not isinstance(lin, nn.Linear):
        return
    with torch.no_grad():
        _init_tiny_diag(lin.weight, scale=scale)
        if lin.bias is not None:
            lin.bias.zero_()

def _normalize_mlp_mode(mode: Optional[str]) -> str:
    if mode in MLP_MODE_SPECS:
        return mode
    if mode == "ann_sub":
        return "mlp_sub"
    if mode == "ann_add":
        return "mlp_add"
    return DEFAULT_MLP_MODE

def _is_substitution_mode(mode: str) -> bool:
    return _normalize_mlp_mode(mode) in {"mlp_sub"}

def _normalize_mlp_arch(arch: Optional[str]) -> str:
    if arch is None:
        return DEFAULT_MLP_ARCH
    arch = str(arch).lower()
    if arch in VALID_MLP_ARCHS:
        return arch
    return DEFAULT_MLP_ARCH

def _is_snn_mode(mode: str) -> bool:
    return MLP_MODE_SPECS.get(mode, {}).get("kind") == "snn"

def _final_activation_for_mode(mode: str, override: Optional[str] = None) -> Tuple[nn.Module, str]:
    if override:
        lowered = override.lower()
        if lowered in {"none", "default"}:
            override = None
        elif lowered in ANN_OUTPUT_ACTIVATIONS:
            ctor, _ = ANN_OUTPUT_ACTIVATIONS[lowered]
            return ctor(), lowered
    spec = MLP_MODE_SPECS.get(mode, MLP_MODE_SPECS[DEFAULT_MLP_MODE])
    default_name = spec.get("activation_name", "sigmoid")
    entry = ANN_OUTPUT_ACTIVATIONS.get(default_name, ANN_OUTPUT_ACTIVATIONS["sigmoid"])
    ctor, _ = entry
    return ctor(), default_name

def _select_mlp_mode(settings: Dict, override: Optional[str] = None) -> str:
    if override is not None:
        return _normalize_mlp_mode(override)
    return _normalize_mlp_mode(settings.get("mlp_mode"))

def _resolve_mlp_hidden_sizes(settings: Dict) -> Optional[List[int]]:
    sizes = settings.get("mlp_hidden_sizes")
    if sizes is None:
        return None
    if isinstance(sizes, (list, tuple)):
        return [int(v) for v in sizes]
    return sizes

class InputCompressionMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_sizes: Optional[List[int]] = None):
        super().__init__()
        sizes = list(hidden_sizes or [])
        layers: List[nn.Module] = []
        prev = int(in_dim)
        for sz in sizes:
            layers.append(nn.Linear(prev, int(sz)))
            layers.append(nn.ReLU())
            prev = int(sz)
        layers.append(nn.Linear(prev, int(out_dim)))
        self.net = nn.Sequential(*layers)
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        if C != self.in_dim:
            raise ValueError(f"InputCompressionMLP expected {self.in_dim} inputs, got {C}.")
        x_flat = x.reshape(B * T, C)
        y = self.net(x_flat)
        return y.view(B, T, self.out_dim)

class ModulatingMLP(nn.Module):
    def __init__(self, nb_inputs, nb_hidden, nb_outputs,
                 hidden_sizes: Optional[List[int]] = None,
                 in_mask: Dict=None, out_mask: Dict=None,
                 mode: str = "mlp_sub", arch: str = "mlp",
                 in_size_override: Optional[Dict[str, int]] = None,
                 out_size_override: Optional[Dict[str, int]] = None,
                 output_activation: Optional[str] = None,
                 nm_cfg: Optional[Dict] = None,
                 nm_target_overrides: Optional[Dict[str, int]] = None,
                 hidden_group_layout: Optional['GroupLayout'] = None,
                 output_group_layout: Optional['GroupLayout'] = None,
                 current_cfg: Optional[Dict] = None,
                 current_target_overrides: Optional[Dict[str, int]] = None):
        super().__init__()
        self.nb_inputs, self.nb_hidden, self.nb_outputs = nb_inputs, nb_hidden, nb_outputs
        self.in_mask = in_mask or {}
        self.out_mask = out_mask or {}
        self.mode = _normalize_mlp_mode(mode)
        self.arch = _normalize_mlp_arch(arch)
        self.stateful = False
        self.output_activation_override = output_activation
        self.output_activation_kind: Optional[str] = None
        self.in_slices, base_out_slices, in_size, out_size = _mlp_io_slices(
            nb_inputs, nb_hidden, nb_outputs, self.in_mask, self.out_mask,
            in_size_override=in_size_override, out_size_override=out_size_override
        )
        self.input_size, self.output_size = in_size, out_size
        sizes = list(hidden_sizes) if hidden_sizes else [2048]
        self.hidden_sizes = sizes
        self.final_activation: Optional[nn.Module] = None
        self.layers: Optional[nn.Sequential] = None
        self.core: Optional[nn.Module] = None
        self.output_layer: Optional[nn.Linear] = None
        self._linear_layers: List[nn.Linear] = []
        self.nm_cfg = nm_cfg or {}
        self.current_cfg = current_cfg or {}
        self.current_mode = bool(self.current_cfg.get("enable", False))
        self.current_out_slices: Dict[str, slice] = {}
        self.current_target = str(self.current_cfg.get("target", "both")).lower()
        self.current_target_overrides = current_target_overrides or {}
        self.hidden_nm_per_neuron = max(0, int(self.nm_cfg.get("hidden_per_neuron", 0)))
        self.output_nm_per_neuron = max(0, int(self.nm_cfg.get("output_per_neuron", 0)))
        self.use_neuromodulators = bool(self.nm_cfg.get("enable")) and (
            self.hidden_nm_per_neuron > 0 or self.output_nm_per_neuron > 0
        )
        self.hidden_group_layout = hidden_group_layout
        self.output_group_layout = output_group_layout
        self.nm_target_overrides = nm_target_overrides or {}
        self.nm_mapper: Optional[NeuromodulatorMapper] = None
        self.nm_out_slices: Dict[str, slice] = {}
        self.hidden_nm_targets = None
        self.output_nm_targets = None
        self.out_slices = base_out_slices
        # NM path: primary outputs raw neuromodulator levels (linear head) so the secondary mapper
        # sees unclamped values. Everything downstream is handled by the secondary MLPs.
        if self.current_mode:
            self.use_neuromodulators = False
        if self.use_neuromodulators and self.mode in {"mlp_sub", "mlp_add"}:
            self.output_activation_override = "linear"
        if self.use_neuromodulators:
            hidden_override = self.nm_target_overrides.get("hidden")
            output_override = self.nm_target_overrides.get("output")
            self.hidden_nm_targets = max(0, int(hidden_override)) if hidden_override is not None else (
                self.hidden_group_layout.group_count if self.hidden_group_layout else nb_hidden
            )
            self.output_nm_targets = max(0, int(output_override)) if output_override is not None else (
                self.output_group_layout.group_count if self.output_group_layout else nb_outputs
            )
            self.nm_mapper = NeuromodulatorMapper(
                self.hidden_nm_targets,
                self.output_nm_targets,
                self.hidden_nm_per_neuron,
                self.output_nm_per_neuron,
                self.nm_cfg.get("init_scale", SOFT_MOD_GAIN_INIT),
                cfg=self.nm_cfg,
            )
            self.out_slices = {}
            offset = 0
            if self.hidden_nm_per_neuron > 0 and self.hidden_nm_targets > 0:
                hid_len = self.nm_mapper.hidden_flat_dim
                self.nm_out_slices["hidden"] = slice(offset, offset + hid_len)
                offset += hid_len
            if self.output_nm_per_neuron > 0 and self.output_nm_targets > 0:
                out_len = self.nm_mapper.output_flat_dim
                self.nm_out_slices["output"] = slice(offset, offset + out_len)
            self.output_size = self.nm_mapper.total_output_dim
        else:
            self.out_slices = base_out_slices

        if self.current_mode:
            current_hidden = self.current_target_overrides.get("hidden", nb_hidden)
            current_output = self.current_target_overrides.get("output", nb_outputs)
            if self.current_target == "hidden":
                current_output = 0
            elif self.current_target == "output":
                current_hidden = 0
            offset = 0
            if current_hidden > 0:
                self.current_out_slices["hidden_current"] = slice(offset, offset + int(current_hidden))
                offset += int(current_hidden)
            if current_output > 0:
                self.current_out_slices["output_current"] = slice(offset, offset + int(current_output))
                offset += int(current_output)
            self.output_size = offset
            self.out_slices = dict(self.current_out_slices)
            current_activation = (self.current_cfg.get("activation") or "").lower()
            if current_activation in {"sigmoid", "tanh"}:
                self.output_activation_override = current_activation

        self._build_mlp_core(sizes)

    def _build_mlp_core(self, sizes: List[int]):
        layers: List[nn.Module] = []
        linear_layers: List[nn.Linear] = []
        in_dim = self.input_size
        for sz in sizes:
            lin = nn.Linear(in_dim, sz)
            # For substitution (mlp_sub), keep activations identity to preserve sign and values.
            act_cls = nn.Identity if self.mode == "mlp_sub" else nn.ReLU
            layers.extend([lin, act_cls()])
            linear_layers.append(lin)
            in_dim = sz
        out_lin = nn.Linear(in_dim, self.output_size)
        final_activation, activation_name = _final_activation_for_mode(self.mode, self.output_activation_override)
        self.output_activation_kind = activation_name
        layers.extend([out_lin, final_activation])
        linear_layers.append(out_lin)
        self.layers = nn.Sequential(*layers)
        self._linear_layers = linear_layers
        if self.mode == "mlp_add":
            self._init_additive_weights_soft()
        elif self.mode == "mlp_sub":
            self._init_partial_identity()
        else:
            self._init_additive_weights()

    def _init_partial_identity(self):
        linear_layers = getattr(self, "_linear_layers", None) or []
        if not linear_layers:
            return
        first, last = linear_layers[0], linear_layers[-1]
        # For ANN substitution, default to full passthrough on all overlapping dims.
        K = min(first.in_features, first.out_features, last.in_features, last.out_features)
        with torch.no_grad():
            for lin in (first, last):
                lin.weight.zero_()
                if lin.bias is not None:
                    lin.bias.zero_()
            if K > 0:
                eyeK = torch.eye(K, dtype=first.weight.dtype, device=first.weight.device)
                first.weight[:K, :K].copy_(eyeK)
                last.weight[:K, :K].copy_(eyeK)

    def _init_additive_weights(self, scale: float = TINY_MOD_INIT_SCALE):
        linear_layers = getattr(self, "_linear_layers", None) or []
        if not linear_layers:
            return
        for lin in linear_layers:
            _init_tiny_linear(lin, scale=scale)

    def _init_additive_weights_soft(self, hidden_scale: float = SOFT_MOD_INIT_SCALE, final_std: float = ANN_ADD_FINAL_STD):
        """
        Soft-start for additive modulation: tiny weights in hidden layers, very small random
        output layer with zero bias so initial deltas are near-zero but not silent.
        """
        linear_layers = getattr(self, "_linear_layers", None) or []
        if not linear_layers:
            return
        # Hidden layers stay tiny/near-identity
        for lin in linear_layers[:-1]:
            _init_tiny_linear(lin, scale=hidden_scale)
        # Final layer gets a small random normal; bias zeroed for neutrality
        last = linear_layers[-1]
        with torch.no_grad():
            last.weight.normal_(mean=0.0, std=final_std)
            if last.bias is not None:
                last.bias.zero_()

    def forward(self, x):
        return self.layers(x)


class SpikingLinearLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        time_step: float,
        tau_syn: float,
        tau_mem: float,
        weight_scale: float = SOFT_MOD_INIT_SCALE,
        recurrent: bool = False,
        param_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight_scale = weight_scale
        self.recurrent = recurrent
        self.param_ranges = param_ranges or PARAMETER_RANGE_DEFAULTS

        self.weight = nn.Parameter(torch.zeros(in_dim, out_dim))
        self.alpha = nn.Parameter(torch.full((1, out_dim), float(np.exp(-time_step / max(tau_syn, 1e-6)))))
        self.beta = nn.Parameter(torch.full((1, out_dim), float(np.exp(-time_step / max(tau_mem, 1e-6)))))
        self.threshold = nn.Parameter(torch.empty((1, out_dim)))
        self.reset = nn.Parameter(torch.empty((1, out_dim)))
        self.rest = nn.Parameter(torch.empty((1, out_dim)))
        if recurrent:
            self.rec_weight = nn.Parameter(torch.zeros(out_dim, out_dim))
        else:
            self.register_parameter("rec_weight", None)

        self._init_parameters()

    def _init_parameters(self):
        with torch.no_grad():
            _init_tiny_diag(self.weight, scale=self.weight_scale)
            if self.rec_weight is not None:
                _init_tiny_diag(self.rec_weight, scale=self.weight_scale)
            thr_lo, thr_hi = self.param_ranges.get("thr", (0.5, 1.5))
            reset_lo, reset_hi = self.param_ranges.get("reset", (-0.5, 0.5))
            rest_lo, rest_hi = self.param_ranges.get("rest", (-0.5, 0.5))
            self.threshold.uniform_(float(thr_lo), float(thr_hi))
            self.reset.uniform_(float(reset_lo), float(reset_hi))
            self.rest.uniform_(float(rest_lo), float(rest_hi))

    def zero_state(self, batch_size: int, device, dtype) -> Dict[str, torch.Tensor]:
        zeros = torch.zeros((batch_size, self.out_dim), device=device, dtype=dtype)
        return {"syn": zeros.clone(), "mem": zeros.clone(), "spikes": zeros.clone()}

    def forward_step(self, inputs: torch.Tensor, state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        h = torch.matmul(inputs, self.weight)
        if self.recurrent and self.rec_weight is not None:
            h = h + torch.matmul(state["spikes"], self.rec_weight)

        alpha = self.alpha.expand_as(state["syn"])
        beta = self.beta.expand_as(state["mem"])
        thr = self.threshold.expand_as(state["mem"])
        rst = self.reset.expand_as(state["mem"])
        rest = self.rest.expand_as(state["mem"])

        mthr = state["mem"] - thr
        spk = spike_fn(mthr)
        rst_mask = (mthr > 0).float()

        syn = alpha * state["syn"] + h
        mem = beta * (state["mem"] - rest) + rest + (1 - beta) * syn - rst_mask * (thr - rst)
        return spk, {"syn": syn, "mem": mem, "spikes": spk}


class NonSpikingLinearLayer(nn.Module):
    """
    Linear layer with synaptic and membrane filtering but no spiking/thresholding.
    Used for parameter readouts where the membrane itself encodes the value.
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        time_step: float,
        tau_syn: float,
        tau_mem: float,
        weight_scale: float = SOFT_MOD_INIT_SCALE,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight_scale = weight_scale

        self.weight = nn.Parameter(torch.zeros(in_dim, out_dim))
        self.alpha = nn.Parameter(torch.full((1, out_dim), float(np.exp(-time_step / max(tau_syn, 1e-6)))))
        self.beta = nn.Parameter(torch.full((1, out_dim), float(np.exp(-time_step / max(tau_mem, 1e-6)))))
        self._init_parameters()

    def _init_parameters(self):
        with torch.no_grad():
            _init_tiny_diag(self.weight, scale=self.weight_scale)

    def zero_state(self, batch_size: int, device, dtype) -> Dict[str, torch.Tensor]:
        zeros = torch.zeros((batch_size, self.out_dim), device=device, dtype=dtype)
        return {"syn": zeros.clone(), "mem": zeros.clone()}

    def forward_step(self, inputs: torch.Tensor, state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        h = torch.matmul(inputs, self.weight)
        alpha = self.alpha.expand_as(state["syn"])
        beta = self.beta.expand_as(state["mem"])
        syn = alpha * state["syn"] + h
        baseline = state.get("baseline")
        if baseline is None:
            baseline = torch.zeros_like(state["mem"])
        # Keep input influence decoupled from the decay-to-baseline term:
        # - With no input, mem decays toward `baseline` with rate beta.
        # - With input, syn is added directly (not scaled by (1-beta)).
        mem = beta * (state["mem"] - baseline) + baseline + syn
        return mem, {"syn": syn, "mem": mem, "baseline": baseline}



class SNNAdditiveModulator(nn.Module):
    """Secondary SNN ("Double" architecture) with optional hidden spiking layers."""

    def __init__(
        self,
        nb_inputs: int,
        nb_hidden: int,
        nb_outputs: int,
        time_step: float,
        tau_syn: float,
        tau_mem: float,
        hidden_layer_sizes: Optional[List[int]] = None,
        hidden_recurrent: bool = False,
        weight_scale: float = SOFT_MOD_INIT_SCALE,
        gain_init: float = SOFT_MOD_GAIN_INIT,
        nm_cfg: Optional[Dict] = None,
        param_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        hidden_group_layout: Optional['GroupLayout'] = None,
        output_group_layout: Optional['GroupLayout'] = None,
        param_hidden_dim: Optional[int] = None,
        param_output_dim: Optional[int] = None,
        spike_hidden_dim: Optional[int] = None,
        spike_output_dim: Optional[int] = None,
        effect_hidden_dim: Optional[int] = None,
        effect_output_dim: Optional[int] = None,
        balanced_init: bool = False,
        init_effect_frac: float = 0.05,
    ):
        super().__init__()
        self.nb_inputs = nb_inputs
        self.nb_hidden = nb_hidden
        self.nb_outputs = nb_outputs
        self.time_step = time_step
        self.tau_syn = tau_syn
        self.tau_mem = tau_mem
        self.hidden_layer_sizes = list(hidden_layer_sizes or [])
        self.hidden_recurrent = hidden_recurrent
        self.weight_scale = weight_scale
        self._param_dims: Dict[str, int] = {}
        self.param_ranges = param_ranges or PARAMETER_RANGE_DEFAULTS
        self.nm_cfg = nm_cfg or {}
        self.hidden_nm_per_neuron = max(0, int(self.nm_cfg.get("hidden_per_neuron", 0)))
        self.output_nm_per_neuron = max(0, int(self.nm_cfg.get("output_per_neuron", 0)))
        self.use_neuromodulators = bool(self.nm_cfg.get("enable")) and (
            self.hidden_nm_per_neuron > 0 or self.output_nm_per_neuron > 0
        )
        self.nm_init_scale = float(self.nm_cfg.get("init_scale", gain_init))
        self.shared_mod_gain: Optional[nn.ParameterDict] = None
        self.balanced_init = bool(balanced_init)
        self.init_effect_frac = float(init_effect_frac)

        self.hidden_group_layout = hidden_group_layout
        self.output_group_layout = output_group_layout
        hidden_effect = self.hidden_group_layout.group_count if self.hidden_group_layout else nb_hidden
        output_effect = self.output_group_layout.group_count if self.output_group_layout else nb_outputs
        if effect_hidden_dim is not None:
            hidden_effect = int(effect_hidden_dim)
        if effect_output_dim is not None:
            output_effect = int(effect_output_dim)
        self.hidden_effect_dim = max(0, int(hidden_effect))
        self.output_effect_dim = max(0, int(output_effect))

        self.param_hidden_dim = max(0, int(param_hidden_dim)) if param_hidden_dim is not None else nb_hidden
        self.param_output_dim = max(0, int(param_output_dim)) if param_output_dim is not None else nb_outputs
        self.spike_hidden_dim = max(0, int(spike_hidden_dim)) if spike_hidden_dim is not None else nb_hidden
        self.spike_output_dim = max(0, int(spike_output_dim)) if spike_output_dim is not None else nb_outputs

        self.feature_block = 5 * self.param_hidden_dim + 2 * self.param_output_dim
        self.effect_block = 5 * self.hidden_effect_dim + 2 * self.output_effect_dim
        self.param_block = self.feature_block
        self.input_dim = nb_inputs + self.spike_hidden_dim + self.spike_output_dim + 1 + self.param_block
        self.nm_mapper: Optional[NeuromodulatorMapper] = None
        if self.use_neuromodulators:
            self.nm_mapper = NeuromodulatorMapper(
                self.hidden_effect_dim,
                self.output_effect_dim,
                self.hidden_nm_per_neuron,
                self.output_nm_per_neuron,
                self.nm_init_scale,
                cfg=self.nm_cfg,
            )
            self.output_dim = self.nm_mapper.total_output_dim
        else:
            self.output_dim = 2 * self.effect_block

        layer_dims = [self.input_dim] + self.hidden_layer_sizes + [self.output_dim]
        self.layers = nn.ModuleList()
        for idx in range(len(layer_dims) - 1):
            in_dim = layer_dims[idx]
            out_dim = layer_dims[idx + 1]
            is_hidden = idx < len(layer_dims) - 2
            recurrent = self.hidden_recurrent and is_hidden
            self.layers.append(
                SpikingLinearLayer(
                    in_dim,
                    out_dim,
                    time_step=time_step,
                    tau_syn=tau_syn,
                    tau_mem=tau_mem,
                    weight_scale=weight_scale,
                    recurrent=recurrent,
                    param_ranges=self.param_ranges,
                )
            )

        self.mod_hidden = self.layers[-1].out_dim
        if self.use_neuromodulators:
            self._build_neuromodulators()
            self.mod_factors = None
            self.shared_mod_gain = None
        else:
            self._build_modulation_slices()
            factors = torch.full((1, self.mod_hidden), float(max(gain_init, 0.0)))
            if gain_init > 0:
                nn.init.normal_(factors, mean=float(gain_init), std=float(gain_init))
            self.mod_factors = nn.Parameter(factors)
            if self.balanced_init:
                self._apply_balanced_init()
            self.shared_mod_gain = None

    def _apply_balanced_init(self):
        if self.mod_factors is None:
            return
        if not hasattr(self, "_mod_slices"):
            return
        out_layer = self.layers[-1] if self.layers else None
        if not isinstance(out_layer, SpikingLinearLayer):
            return
        with torch.no_grad():
            # Force paired positive/negative neurons to be identical so effects cancel at init.
            for _, (start, mid, end) in self._mod_slices.items():
                if end <= mid or mid <= start:
                    continue
                out_layer.weight[:, mid:end] = out_layer.weight[:, start:mid]
                out_layer.alpha[:, mid:end] = out_layer.alpha[:, start:mid]
                out_layer.beta[:, mid:end] = out_layer.beta[:, start:mid]
                out_layer.threshold[:, mid:end] = out_layer.threshold[:, start:mid]
                out_layer.reset[:, mid:end] = out_layer.reset[:, start:mid]
                out_layer.rest[:, mid:end] = out_layer.rest[:, start:mid]

            # Initialize gains so a 1-spike imbalance corresponds to ~init_effect_frac of each param's range width.
            frac = max(0.0, float(self.init_effect_frac))
            for name, (start, mid, end) in self._mod_slices.items():
                lo, hi = self.param_ranges.get(name, PARAMETER_RANGE_DEFAULTS.get(name, (-1.0, 1.0)))
                width = float(hi) - float(lo)
                gain = frac * width
                self.mod_factors.data[:, start:mid] = gain
                self.mod_factors.data[:, mid:end] = gain

    def _build_modulation_slices(self):
        self._mod_slices: Dict[str, Tuple[int, int, int]] = {}
        offset = 0
        for name, dim in [
            ("alpha_1", self.hidden_effect_dim),
            ("beta_1", self.hidden_effect_dim),
            ("thr", self.hidden_effect_dim),
            ("reset", self.hidden_effect_dim),
            ("rest", self.hidden_effect_dim),
            ("alpha_2", self.output_effect_dim),
            ("beta_2", self.output_effect_dim),
        ]:
            start = offset
            mid = start + dim
            end = mid + dim
            self._mod_slices[name] = (start, mid, end)
            self._param_dims[name] = dim
            offset = end
        if offset != self.mod_hidden:
            raise ValueError("Modulation slice construction failed: mismatched neuron count.")

    def zero_state(self, batch_size: int, init_params: Optional[Dict[str, torch.Tensor]] = None) -> List[Dict[str, torch.Tensor]]:
        device = self.layers[0].weight.device
        dtype = self.layers[0].weight.dtype
        states = [layer.zero_state(batch_size, device, dtype) for layer in self.layers]
        return states

    def build_features(
        self,
        alpha_1: torch.Tensor,
        beta_1: torch.Tensor,
        thr: torch.Tensor,
        reset: torch.Tensor,
        rest: torch.Tensor,
        alpha_2: torch.Tensor,
        beta_2: torch.Tensor,
        input_t: torch.Tensor,
        hidden_spikes: torch.Tensor,
        readout_spikes: torch.Tensor,
        step_idx: int,
    ) -> torch.Tensor:
        param_stack = torch.cat(
            [alpha_1, beta_1, thr, reset, rest, alpha_2, beta_2],
            dim=1
        )
        time_feat = torch.full((input_t.size(0), 1), float(step_idx), device=input_t.device, dtype=input_t.dtype)
        return torch.cat(
            [param_stack, time_feat, input_t, hidden_spikes, readout_spikes],
            dim=1
        )

    def forward_step(
        self,
        features: torch.Tensor,
        states: List[Dict[str, torch.Tensor]]
    ) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
        x = features
        new_states = []
        for layer, state in zip(self.layers, states):
            spk, next_state = layer.forward_step(x, state)
            new_states.append(next_state)
            x = spk
        return x, new_states

    def _build_neuromodulators(self):
        # NeuromodulatorMapper is constructed in __init__; nothing to do here.
        return

    def modulation_effects(self, spike_accum: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.use_neuromodulators:
            if self.nm_mapper is None:
                return {}
            return self.nm_mapper.effects_from_flat(spike_accum)
        effects = {}
        for name, (start, mid, end) in self._mod_slices.items():
            gains_pos = self.mod_factors[:, start:mid]
            gains_neg = self.mod_factors[:, mid:end]
            # Element-wise +/- contributions: keep "which neuron spiked" information.
            pos_spikes = spike_accum[:, start:mid]
            neg_spikes = spike_accum[:, mid:end]
            base = pos_spikes * gains_pos - neg_spikes * gains_neg
            effects[name] = base
        return effects

    def clamp_mod_factors_(self):
        if self.mod_factors is None:
            return
        with torch.no_grad():
            self.mod_factors.clamp_(min=0.0)


# =============================================================================
# SECTION 9: Build Modulator
# =============================================================================

def build_modulator(
    settings: Dict,
    override_mode: Optional[str] = None,
    hidden_sizes: Optional[List[int]] = None,
) -> nn.Module:
    mode = _select_mlp_mode(settings, override=override_mode)
    settings["_mod_output_size"] = None
    param_ranges = _resolve_param_ranges(settings)
    substitution_mode = _is_substitution_mode(mode)
    mod_input_size = int(settings.get("nb_inputs_mod", settings["nb_inputs"]))
    smoothing_cfg = settings.get("param_smoothing", {})
    smoothing_modes = {_normalize_mlp_mode(m) for m in smoothing_cfg.get("modes", [])} if smoothing_cfg else set()
    smoothing_allowed = (
        bool(smoothing_cfg.get("enable"))
        and (not smoothing_modes or mode in smoothing_modes)
    )
    hidden_layout = _get_group_layout(settings, 0, settings["nb_hidden"])
    output_layout = _get_group_layout(settings, 1, settings["nb_outputs"])
    mod_mask = _build_fixed_mod_mask(settings, hidden_layout, output_layout)
    if _is_snn_mode(mode):
        snn_hidden_layers = hidden_sizes
        if snn_hidden_layers is None:
            snn_hidden_layers = settings.get("mod_hidden_sizes")
        param_hidden_dim = param_output_dim = None
        spike_hidden_dim = spike_output_dim = None
        effect_hidden_dim = effect_output_dim = None
        if mod_mask is not None:
            nm_cfg = settings.get("nm_cfg", {}) or {}
            nm_active = bool(nm_cfg.get("enable")) and (
                int(nm_cfg.get("hidden_per_neuron", 0)) > 0 or int(nm_cfg.get("output_per_neuron", 0)) > 0
            )
            hidden_union = int(mod_mask["hidden_union_idx"].numel())
            output_union = int(mod_mask["output_union_idx"].numel())
            hidden_neuron = int(mod_mask["hidden_neuron_idx"].numel())
            output_neuron = int(mod_mask["output_neuron_idx"].numel())
            if nm_active:
                effect_hidden_dim = hidden_neuron
                effect_output_dim = output_neuron
            else:
                effect_hidden_dim = hidden_union
                effect_output_dim = output_union
            param_hidden_dim = hidden_union
            param_output_dim = output_union
            if mod_mask.get("flat_inputs") and mod_mask.get("flat_hidden_idx") is not None:
                spike_hidden_dim = int(mod_mask["flat_hidden_idx"].numel())
                spike_output_dim = int(mod_mask["flat_output_idx"].numel()) if mod_mask.get("flat_output_idx") is not None else None
        mod = SNNAdditiveModulator(
                nb_inputs=mod_input_size,
                nb_hidden=settings["nb_hidden"],
                nb_outputs=settings["nb_outputs"],
                time_step=settings["time_step"],
                tau_syn=settings.get("tau_syn", 10e-3),
                tau_mem=settings.get("tau_mem", 20e-3),
                hidden_layer_sizes=snn_hidden_layers,
                hidden_recurrent=settings.get("snn_mod_hidden_recurrent", False),
                weight_scale=settings.get("snn_mod_weight_scale", SNN_ADD_WEIGHT_SCALE_DEFAULT),
                gain_init=settings.get("snn_mod_gain_init", SNN_ADD_GAIN_INIT_DEFAULT),
                nm_cfg=settings.get("nm_cfg", {}),
                param_ranges=param_ranges,
                hidden_group_layout=hidden_layout,
                output_group_layout=output_layout,
                param_hidden_dim=param_hidden_dim,
                param_output_dim=param_output_dim,
                spike_hidden_dim=spike_hidden_dim,
                spike_output_dim=spike_output_dim,
                effect_hidden_dim=effect_hidden_dim,
                effect_output_dim=effect_output_dim,
                balanced_init=settings.get("snn_add_balanced_init", False),
                init_effect_frac=settings.get("snn_add_init_effect_frac", 0.05),
            ).to(device)
        if settings.get("snn_mod_hidden_recurrent", False) and settings.get("snn_mod_rec_init_zero", False):
            for layer in getattr(mod, "layers", []):
                rec = getattr(layer, "rec_weight", None)
                if rec is not None:
                    with torch.no_grad():
                        rec.zero_()
        if settings.get("param_timescales", {}).get("enable"):
            mod.param_timescales = ParamTimescaleController(
                settings.get("ann_interval", settings.get("mlp_interval", 3)),
                MOD_TARGET_PARAM_NAMES,
                dist=settings.get("param_timescales", {}).get("distribution", "fixed"),
                scale=settings.get("param_timescales", {}).get("scale", 0.0),
                std=settings.get("param_timescales", {}).get("std", 0.0),
                seed=settings.get("param_timescales", {}).get("seed"),
                trainable=settings.get("param_timescales", {}).get("trainable", True),
            ).to(device)
        if smoothing_allowed:
            mod.param_smoothing = ParamSmoothingController(
                MOD_TARGET_PARAM_NAMES,
                settings["nb_hidden"],
                settings["nb_outputs"],
                tau_init=smoothing_cfg.get("tau_init", PARAM_SMOOTH_TAU_INIT_DEFAULT),
                tau_min=smoothing_cfg.get("tau_min", PARAM_SMOOTH_TAU_MIN_DEFAULT),
                trainable=smoothing_cfg.get("trainable", True),
            ).to(device)
        if settings.get("channel_compress_mode") == "mod_mlp" and settings.get("channel_compress_enable"):
            hidden_sizes = settings.get("channel_compress_mlp_hidden_sizes") or []
            mod.input_compressor = InputCompressionMLP(
                settings["nb_inputs"], mod_input_size, hidden_sizes=hidden_sizes
            ).to(device)
        settings["snn_mod_hidden"] = mod.mod_hidden
        return mod
    resolved_hidden = hidden_sizes if hidden_sizes is not None else _resolve_mlp_hidden_sizes(settings)
    additive_mode = (mode == "mlp_add")
    out_size_override = _ann_out_size_overrides(settings)
    in_size_override = _ann_in_size_overrides(settings, substitution_mode=substitution_mode)
    nm_target_overrides = None
    current_target_overrides = None
    mod_hid_flat_group = bool(settings.get("mod_hid_flat_group", False))
    mod_hid_flat_mod_only = bool(settings.get("mod_hid_flat_modulated_only", False))
    if mod_mask is not None:
        param_counts = {}
        for name, idx in mod_mask.get("hidden_param_idx", {}).items():
            param_counts[name] = int(idx.numel())
        for name, idx in mod_mask.get("output_param_idx", {}).items():
            param_counts[name] = int(idx.numel())
        if param_counts:
            out_size_override = dict(out_size_override)
            in_size_override = dict(in_size_override)
            out_size_override.update(param_counts)
            in_size_override.update(param_counts)
        if mod_mask.get("flat_inputs"):
            if mod_mask.get("flat_hidden_idx") is not None and not (mod_hid_flat_group or mod_hid_flat_mod_only):
                in_size_override["hid_flat"] = int(mod_mask["flat_hidden_idx"].numel())
            if mod_mask.get("flat_output_idx") is not None:
                in_size_override["out_flat"] = int(mod_mask["flat_output_idx"].numel())
        nm_target_overrides = {
            "hidden": int(mod_mask["hidden_neuron_idx"].numel()),
            "output": int(mod_mask["output_neuron_idx"].numel()),
        }
        current_target_overrides = {
            "hidden": int(mod_mask["hidden_union_idx"].numel()),
            "output": int(mod_mask["output_union_idx"].numel()),
        }
    hid_flat_dim = None
    if mod_hid_flat_group:
        hid_flat_dim = hidden_layout.group_count if hidden_layout else settings["nb_hidden"]
    if mod_hid_flat_mod_only and mod_mask is not None:
        if mod_hid_flat_group:
            hid_flat_dim = int(mod_mask["hidden_union_idx"].numel())
        else:
            mod_only_count = _count_modulated_neurons(
                hidden_layout,
                mod_mask.get("hidden_union_idx"),
                settings["nb_hidden"],
            )
            if mod_only_count is not None:
                hid_flat_dim = int(mod_only_count)
    if hid_flat_dim is not None:
        in_size_override = dict(in_size_override)
        in_size_override["hid_flat"] = int(hid_flat_dim)
    output_activation = settings.get("ann_output_activation") or settings.get("mlp_output_activation")
    hidden_nm_layout = _uniform_group_layout(hidden_layout) if substitution_mode else hidden_layout
    output_nm_layout = _uniform_group_layout(output_layout) if substitution_mode else output_layout
    current_cfg = None
    if settings.get("mod_current_enable"):
        current_cfg = {
            "enable": True,
            "target": settings.get("mod_current_target", "both"),
            "activation": settings.get("mod_current_activation", "tanh"),
        }
    ann_mod = ModulatingMLP(
        mod_input_size, settings["nb_hidden"], settings["nb_outputs"],
        hidden_sizes=resolved_hidden,
        in_mask=settings.get("mlp_in_mask", {}),
        out_mask=settings.get("mlp_out_mask", {}),
        mode=mode,
        arch=settings.get("mlp_arch", DEFAULT_MLP_ARCH),
        in_size_override=in_size_override,
        out_size_override=out_size_override,
        output_activation=output_activation,
        nm_cfg=settings.get("nm_cfg", {}),
        nm_target_overrides=nm_target_overrides,
        hidden_group_layout=hidden_nm_layout,
        output_group_layout=output_nm_layout,
        current_cfg=current_cfg,
        current_target_overrides=current_target_overrides,
    ).to(device)
    settings["_mod_output_size"] = int(getattr(ann_mod, "output_size", 0))
    if settings.get("param_timescales", {}).get("enable"):
        ann_mod.param_timescales = ParamTimescaleController(
            settings.get("ann_interval", settings.get("mlp_interval", 3)),
            MOD_TARGET_PARAM_NAMES,
            dist=settings.get("param_timescales", {}).get("distribution", "fixed"),
            scale=settings.get("param_timescales", {}).get("scale", 0.0),
            std=settings.get("param_timescales", {}).get("std", 0.0),
            seed=settings.get("param_timescales", {}).get("seed"),
            trainable=settings.get("param_timescales", {}).get("trainable", True),
        ).to(device)
    if smoothing_allowed:
        ann_mod.param_smoothing = ParamSmoothingController(
            MOD_TARGET_PARAM_NAMES,
            settings["nb_hidden"],
            settings["nb_outputs"],
            tau_init=smoothing_cfg.get("tau_init", PARAM_SMOOTH_TAU_INIT_DEFAULT),
            tau_min=smoothing_cfg.get("tau_min", PARAM_SMOOTH_TAU_MIN_DEFAULT),
            trainable=smoothing_cfg.get("trainable", True),
        ).to(device)
    if settings.get("channel_compress_mode") == "mod_mlp" and settings.get("channel_compress_enable"):
        hidden_sizes = settings.get("channel_compress_mlp_hidden_sizes") or []
        ann_mod.input_compressor = InputCompressionMLP(
            settings["nb_inputs"], mod_input_size, hidden_sizes=hidden_sizes
        ).to(device)
    return ann_mod

# =============================================================================
# SECTION 10: Modulated SNN Forward Pass
# =============================================================================
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

    inputs_mod = inputs
    compress_mode = settings.get("channel_compress_mode")
    if compress_mode == "mod_only":
        factor = int(settings.get("channel_compress_factor_mod", settings.get("channel_compress_factor", 1)))
        if factor > 1:
            inputs_mod = compress_dense_inputs(inputs, factor, settings.get("nb_inputs_mod"))
    elif compress_mode == "mod_mlp":
        compressor = getattr(modulator, "input_compressor", None)
        if compressor is None:
            raise RuntimeError("channel_compress_mode=mod_mlp requires an input compressor on the modulator.")
        inputs_mod = compressor(inputs)
    h1_in = torch.einsum("btc,cd->btd", inputs, w1)  # [B,T,H]

    flt2 = torch.zeros((batch_size, nb_outputs), device=device, dtype=dtype)
    out2 = torch.zeros((batch_size, nb_outputs), device=device, dtype=dtype)
    out_rec = []

    mem_rec, spk_rec = [], []
    in_block_mask = mlp_input_block_mask or {}
    out_block_mask = mlp_output_block_mask or {}
    mlp_mode = _select_mlp_mode(settings)
    snn_mode = _is_snn_mode(mlp_mode)
    snn_add_mode = snn_mode and (mlp_mode == "snn_add")
    if snn_mode and not isinstance(modulator, SNNAdditiveModulator):
        raise TypeError("snn mode requires an SNNAdditiveModulator instance.")
    mlp = modulator if not snn_mode else None
    additive_mode = (mlp_mode == "mlp_add")
    if mlp is not None and hasattr(mlp, "reset_sequence_state"):
        mlp.reset_sequence_state(batch_size, device=device, dtype=dtype)
    mlp_state_each_step = False
    update_every_step = bool(settings.get("mod_update_every_step", False))

    # Small version does not expose ANN combo; keep this false for the shared update path.
    combo_mode = False
    combo_additive = set()
    combo_activation_kind = "tanh"

    if snn_mode:
        mod_hidden = modulator.mod_hidden
        mod_state = modulator.zero_state(batch_size)
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

    nm_debug_print = False
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

# Export model/data helpers used by the separate CLI file.
__all__ = [name for name in globals() if not name.startswith("__")]
