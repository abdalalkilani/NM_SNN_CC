"""Imports, device selection, and shared constants."""
# ===== Block 1: Core (SNN + Modulated SNN) =====

import os, math, random, ast, copy, json
from pathlib import Path
from typing import Any, Dict, Tuple, Union, Optional, List, Callable

import h5py, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F

import argparse

# Device / dtype
dtype = torch.float
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

SNN_PARAM_NAMES = [
    "w1", "w2", "v1",
    "alpha_hetero_1", "beta_hetero_1",
    "alpha_hetero_2", "beta_hetero_2",
    "thresholds_1", "reset_1", "rest_1",
    "input_delay_logits",
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




__all__ = [name for name in globals() if not name.startswith('__')]
