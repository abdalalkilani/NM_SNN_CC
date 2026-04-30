"""Refactored package for the unified SNN / modulated SNN runner."""

from .cli import main
from .snn import setup_model, run_snn_hetero, eval_val_metrics, evaluate_testset, train_snn_hetero
from .modulation import build_modulator, ModulatingMLP, SNNAdditiveModulator, SNNSubstitutionModulator
from .modulated import (
    load_base_snn_state,
    load_modulated_checkpoint,
    pick_random_stockpile_ckpt,
    run_snn_modulated,
    train_modulated_snn,
    maybe_run_ann_shap,
)

__all__ = [
    "main",
    "setup_model",
    "run_snn_hetero",
    "eval_val_metrics",
    "evaluate_testset",
    "train_snn_hetero",
    "build_modulator",
    "ModulatingMLP",
    "SNNAdditiveModulator",
    "SNNSubstitutionModulator",
    "load_base_snn_state",
    "load_modulated_checkpoint",
    "pick_random_stockpile_ckpt",
    "run_snn_modulated",
    "train_modulated_snn",
    "maybe_run_ann_shap",
]
