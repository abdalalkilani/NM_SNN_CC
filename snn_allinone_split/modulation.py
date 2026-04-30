"""ANN and SNN modulator models plus modulator construction."""
from .snn import *
from .reporting import *

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
VALID_MLP_ARCHS = {"mlp", "rnn", "lstm"}
MLP_MODE_SPECS = {
    "mlp_sub": {"activation": nn.Sigmoid, "activation_name": "sigmoid", "kind": "mlp"},
    "mlp_add": {"activation": nn.Tanh, "activation_name": "tanh", "kind": "mlp"},
    "mlp_combo": {"activation": nn.Tanh, "activation_name": "tanh", "kind": "mlp"},
    "snn_add": {"activation": nn.Identity, "activation_name": "linear", "kind": "snn"},
    "snn_sub": {"activation": nn.Identity, "activation_name": "linear", "kind": "snn"},
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
    if mode == "ann_combo":
        return "mlp_combo"
    return DEFAULT_MLP_MODE

def _is_substitution_mode(mode: str) -> bool:
    return _normalize_mlp_mode(mode) in {"mlp_sub", "snn_sub"}

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
        self.stateful = self.arch in {"rnn", "lstm"}
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
        self.rnn_hidden_size: Optional[int] = None
        self.rnn_num_layers: Optional[int] = None
        self._state = None
        self._state_batch = None
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
        if self.use_neuromodulators and self.mode in {"mlp_sub", "mlp_add", "mlp_combo"}:
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

        if self.arch == "mlp":
            self._build_mlp_core(sizes)
        else:
            self._build_recurrent_core(sizes)

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
        elif self.mode == "mlp_combo":
            self._init_additive_weights_soft()
        elif self.mode == "mlp_sub":
            self._init_partial_identity()
        else:
            self._init_additive_weights()

    def _build_recurrent_core(self, sizes: List[int]):
        if not sizes:
            sizes = [2048]
        hidden_size = int(sizes[0])
        if any(int(sz) != hidden_size for sz in sizes):
            raise ValueError("All hidden sizes must match when using an RNN/LSTM modulator.")
        num_layers = max(1, len(sizes))
        rnn_kwargs = dict(
            input_size=self.input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        if self.arch == "rnn":
            rnn_kwargs["nonlinearity"] = "tanh"
            self.core = nn.RNN(**rnn_kwargs)
        else:
            self.core = nn.LSTM(**rnn_kwargs)
        self.output_layer = nn.Linear(hidden_size, self.output_size)
        self._linear_layers = [self.output_layer]
        self.rnn_hidden_size = hidden_size
        self.rnn_num_layers = num_layers
        final_activation, activation_name = _final_activation_for_mode(self.mode, self.output_activation_override)
        self.final_activation = final_activation
        self.output_activation_kind = activation_name
        if self.mode == "mlp_add":
            self._init_additive_weights_soft()
        elif self.mode == "mlp_combo":
            self._init_additive_weights_soft()

    def _init_partial_identity(self):
        linear_layers = getattr(self, "_linear_layers", None) or []
        if not linear_layers:
            return
        first, last = linear_layers[0], linear_layers[-1]
        # For substitution (snn_sub / mlp_sub), default to full passthrough on all overlapping dims.
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

    def _zero_state(self, batch_size: int, device, dtype):
        if not self.stateful or batch_size <= 0:
            return None
        layers = self.rnn_num_layers or 1
        hidden = self.rnn_hidden_size or self.output_size
        zeros = torch.zeros(layers, batch_size, hidden, device=device, dtype=dtype)
        if self.arch == "lstm":
            return (zeros.clone(), zeros.clone())
        return zeros

    def reset_sequence_state(self, batch_size: int = 0, device=None, dtype=None):
        if not self.stateful:
            return
        if batch_size <= 0:
            self._state = None
            self._state_batch = None
            return
        ref = self.output_layer if self.output_layer is not None else self._linear_layers[-1]
        dev = device or ref.weight.device
        dt = dtype or ref.weight.dtype
        self._state = self._zero_state(batch_size, dev, dt)
        self._state_batch = batch_size

    def _ensure_state(self, batch_size: int, device, dtype):
        if not self.stateful:
            return None
        if (self._state is None) or (self._state_batch != batch_size):
            self.reset_sequence_state(batch_size, device=device, dtype=dtype)
        return self._state

    def forward(self, x):
        if self.arch == "mlp":
            return self.layers(x)
        batch_size = x.size(0)
        seq = x.unsqueeze(1)
        state = self._ensure_state(batch_size, x.device, x.dtype)
        if self.arch == "lstm":
            out, new_state = self.core(seq, state)
        else:
            out, new_state = self.core(seq, state)
        self._state = new_state
        self._state_batch = batch_size
        logits = self.output_layer(out[:, -1, :])
        if self.final_activation is not None:
            return self.final_activation(logits)
        return logits


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


class SNNSubstitutionModulator(nn.Module):
    """Secondary SNN that directly proposes parameter values via output membrane potentials."""

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
        scale_init: float = SNN_SUB_SCALE_INIT_DEFAULT,
        bias_init: float = SNN_SUB_BIAS_INIT_DEFAULT,
        param_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        nm_cfg: Optional[Dict] = None,
        hidden_group_layout: Optional['GroupLayout'] = None,
        output_group_layout: Optional['GroupLayout'] = None,
        param_hidden_dim: Optional[int] = None,
        param_output_dim: Optional[int] = None,
        spike_hidden_dim: Optional[int] = None,
        spike_output_dim: Optional[int] = None,
        effect_hidden_dim: Optional[int] = None,
        effect_output_dim: Optional[int] = None,
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
        self.scale_init = float(scale_init)
        self.bias_init = float(bias_init)
        self.param_ranges = param_ranges or PARAMETER_RANGE_DEFAULTS
        self._param_dims: Dict[str, int] = {}
        self.nm_cfg = nm_cfg or {}
        self.hidden_nm_per_neuron = max(0, int(self.nm_cfg.get("hidden_per_neuron", 0)))
        self.output_nm_per_neuron = max(0, int(self.nm_cfg.get("output_per_neuron", 0)))
        self.use_neuromodulators = bool(self.nm_cfg.get("enable")) and (
            self.hidden_nm_per_neuron > 0 or self.output_nm_per_neuron > 0
        )
        self.nm_init_scale = float(self.nm_cfg.get("init_scale", weight_scale))
        self.nm_mapper: Optional[NeuromodulatorMapper] = None

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
            self.output_dim = self.effect_block

        layer_dims = [self.input_dim] + self.hidden_layer_sizes + [self.output_dim]
        self.layers = nn.ModuleList()
        for idx in range(len(layer_dims) - 1):
            in_dim = layer_dims[idx]
            out_dim = layer_dims[idx + 1]
            is_hidden = idx < len(layer_dims) - 2
            recurrent = self.hidden_recurrent and is_hidden
            if is_hidden:
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
            else:
                self.layers.append(
                    NonSpikingLinearLayer(
                        in_dim,
                        out_dim,
                        time_step=time_step,
                        tau_syn=tau_syn,
                        tau_mem=tau_mem,
                        weight_scale=weight_scale,
                    )
                )

        self.mod_hidden = self.layers[-1].out_dim
        if not self.use_neuromodulators:
            self._build_output_slices()
            scales: Dict[str, nn.Parameter] = {}
            biases: Dict[str, nn.Parameter] = {}
            for name, dim in self._param_dims.items():
                scales[name] = nn.Parameter(torch.full((1, dim), float(self.scale_init)))
                biases[name] = nn.Parameter(torch.full((1, dim), float(self.bias_init)))
            self.param_scale = nn.ParameterDict(scales)
            self.param_bias = nn.ParameterDict(biases)
            # Prefer "no decay" for direct parameter readout units (they should hold values unless driven).
            out_layer = self.layers[-1]
            if isinstance(out_layer, NonSpikingLinearLayer):
                with torch.no_grad():
                    out_layer.beta.fill_(1.0)
        else:
            self.param_scale = nn.ParameterDict({})
            self.param_bias = nn.ParameterDict({})
        self._init_random_weights_like_primary()

    def _init_random_weights_like_primary(self):
        # For snn_sub, prefer a random init closer to the primary SNN rather than a tiny diagonal.
        # Keep this localized to SNNSubstitutionModulator so snn_add behavior is unchanged.
        with torch.no_grad():
            for layer in self.layers:
                w = getattr(layer, "weight", None)
                if isinstance(w, torch.Tensor) and w.dim() == 2:
                    in_dim = max(1, int(w.size(0)))
                    # Use at least ~primary-SNN-scale init unless user explicitly increases it.
                    std = float(max(self.weight_scale, 2e-2))
                    nn.init.normal_(w, mean=0.0, std=std / math.sqrt(in_dim))
                rec = getattr(layer, "rec_weight", None)
                if isinstance(rec, torch.Tensor) and rec.dim() == 2:
                    d = max(1, int(rec.size(0)))
                    std = float(max(self.weight_scale, 2e-2))
                    nn.init.normal_(rec, mean=0.0, std=std / math.sqrt(d))

    def _build_output_slices(self):
        self._out_slices: Dict[str, Tuple[int, int]] = {}
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
            end = start + dim
            self._out_slices[name] = (start, end)
            self._param_dims[name] = dim
            offset = end
        if offset != self.output_dim:
            raise ValueError("Substitution slice construction failed: mismatched neuron count.")

    def modulation_effects(self, output_mem: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.use_neuromodulators and self.nm_mapper is not None:
            return self.nm_mapper.effects_from_flat(output_mem)
        return self.decode_parameters(output_mem, self.param_ranges)

    def zero_state(self, batch_size: int, init_params: Optional[Dict[str, torch.Tensor]] = None) -> List[Dict[str, torch.Tensor]]:
        device = self.layers[0].weight.device
        dtype = self.layers[0].weight.dtype
        states = [layer.zero_state(batch_size, device, dtype) for layer in self.layers]
        # Initialize output membrane so decoded parameters match the current SNN parameters.
        # Also store a baseline so output units decay back to the init values when un-driven.
        out_state = states[-1]
        if "mem" in out_state:
            if self.use_neuromodulators:
                out_state["mem"] = out_state["mem"].zero_()
            else:
                base_mem = torch.zeros((batch_size, self.output_dim), device=device, dtype=dtype)
                if init_params:
                    eps = 1e-6
                    for name, (start, end) in self._out_slices.items():
                        tensor = init_params.get(name)
                        if tensor is None:
                            continue
                        lo, hi = self.param_ranges.get(name, PARAMETER_RANGE_DEFAULTS.get(name, (-1.0, 1.0)))
                        lo_f = float(lo); hi_f = float(hi)
                        mid = 0.5 * (lo_f + hi_f)
                        half = 0.5 * (hi_f - lo_f)
                        if half <= 0:
                            base_mem[:, start:end] = 0.0
                            continue
                        val = tensor.to(device=device, dtype=dtype)
                        norm = (val - mid) / half
                        norm = torch.clamp(norm, min=-1.0 + eps, max=1.0 - eps)
                        z0 = torch.atanh(norm)
                        scale = self.param_scale[name].to(device=device, dtype=dtype)
                        bias = self.param_bias[name].to(device=device, dtype=dtype)
                        scale_safe = torch.where(scale.abs() < eps, scale.new_full(scale.shape, eps), scale)
                        base_mem[:, start:end] = (z0 - bias) / scale_safe
                out_state["mem"] = base_mem
                out_state["baseline"] = base_mem.clone()
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

    def decode_parameters(self, mem_values: torch.Tensor, param_ranges: Dict[str, Tuple[float, float]]) -> Dict[str, torch.Tensor]:
        effects: Dict[str, torch.Tensor] = {}
        for name, (start, end) in self._out_slices.items():
            raw = mem_values[:, start:end]
            scale = self.param_scale[name].to(device=raw.device, dtype=raw.dtype)
            bias = self.param_bias[name].to(device=raw.device, dtype=raw.dtype)
            lo, hi = param_ranges.get(name, PARAMETER_RANGE_DEFAULTS.get(name, (-float("inf"), float("inf"))))
            lo_f = float(lo); hi_f = float(hi)
            mid = 0.5 * (lo_f + hi_f)
            half = 0.5 * (hi_f - lo_f)
            z = scale * raw + bias
            # Soft-bounded mapping: diminishing effect near boundaries without hard clamping.
            effects[name] = mid + half * torch.tanh(z)
        return effects


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
        if mode == "snn_sub":
            mod = SNNSubstitutionModulator(
                nb_inputs=mod_input_size,
                nb_hidden=settings["nb_hidden"],
                nb_outputs=settings["nb_outputs"],
                time_step=settings["time_step"],
                tau_syn=settings.get("tau_syn", 10e-3),
                tau_mem=settings.get("tau_mem", 20e-3),
                hidden_layer_sizes=snn_hidden_layers,
                hidden_recurrent=settings.get("snn_mod_hidden_recurrent", False),
                weight_scale=settings.get("snn_mod_weight_scale", SNN_ADD_WEIGHT_SCALE_DEFAULT),
                scale_init=settings.get("snn_sub_scale_init", SNN_SUB_SCALE_INIT_DEFAULT),
                bias_init=settings.get("snn_sub_bias_init", SNN_SUB_BIAS_INIT_DEFAULT),
                param_ranges=param_ranges,
                nm_cfg=settings.get("nm_cfg", {}),
                hidden_group_layout=hidden_layout,
                output_group_layout=output_layout,
                param_hidden_dim=param_hidden_dim,
                param_output_dim=param_output_dim,
                spike_hidden_dim=spike_hidden_dim,
                spike_output_dim=spike_output_dim,
                effect_hidden_dim=effect_hidden_dim,
                effect_output_dim=effect_output_dim,
            ).to(device)
        else:
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

# -------------------------
# Load base SNN params from ckpt (for mod run)
# -------------------------


__all__ = [name for name in globals() if not name.startswith('__')]
