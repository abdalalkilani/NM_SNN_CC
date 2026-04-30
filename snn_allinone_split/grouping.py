"""Neuromodulator, grouping, fixed-mask, and parameter controller helpers."""
from .utils import *

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




__all__ = [name for name in globals() if not name.startswith('__')]
