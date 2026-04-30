"""Command-line interface for the refactored SNN runner."""
import argparse
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .modulated import *

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
    p.add_argument("--use_input_delay", type=str2bool, nargs="?", const=True, default=False,
                   help="Enable learnable per-input delay kernel between inputs and hidden layer.")
    p.add_argument("--input_delay_steps", type=int, default=1,
                   help="Max causal delay window (steps) per input; used only when --use_input_delay is True.")
    p.add_argument("--input_delay_init", type=int, default=None,
                   help="Optional fixed initial delay index; if omitted, random per-input init within cap.")
    p.add_argument("--input_delay_init_cap", type=int, default=None,
                   help="Cap for random initial delay (0..cap); defaults to input_delay_steps when omitted.")
    p.add_argument("--input_delay_init_bias", type=float, default=0.1,
                   help="Small bias added to the chosen initial delay logit.")
    p.add_argument("--input_delay_init_noise", type=float, default=0.01,
                   help="Stddev of Gaussian noise added to delay logits at init.")
    p.add_argument("--input_delay_temp", type=float, default=1.0,
                   help="Softmax temperature for delay selection (straight-through argmax uses this temp).")
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
                   help="Enable spike regularization term for the modulator SNN (snn_add/snn_sub). Default=False.")
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
                   choices=["ann_sub", "ann_add", "ann_combo", "snn_add", "snn_sub"],
                   help="Modulation strategy: ann_sub (~mlp_sub), ann_add (~mlp_add), ann_combo (~mlp_combo), snn_add, or snn_sub.")
    p.add_argument("--ann_combo_additive", type=str, nargs="+", default=None,
                   help="Comma-separated or JSON list of params treated additively in ann_combo (default: thr, reset, rest).")
    p.add_argument("--ann_combo_multiplicative", type=str, nargs="+", default=None,
                   help="Comma-separated or JSON list of params treated multiplicatively in ann_combo (default: remaining params).")
    p.add_argument("--ann_arch", type=str, default="mlp", choices=["mlp", "rnn", "lstm"],
                   help="Backbone used for ann_sub/ann_add modulators.")
    p.add_argument("--ann_rnn_state_every_step", type=str2bool, nargs="?", const=True, default=False,
                   help="If True and ann_arch is rnn/lstm, advance recurrent state every timestep even when ann_interval>1 (updates still gated by interval).")
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
    p.add_argument("--snn_sub_scale_init", type=float, default=SNN_SUB_SCALE_INIT_DEFAULT,
                   help="Initial multiplicative factor applied to snn_sub membrane potentials before squashing to parameter ranges.")
    p.add_argument("--snn_sub_bias_init", type=float, default=SNN_SUB_BIAS_INIT_DEFAULT,
                   help="Initial additive bias applied to snn_sub membrane potentials before squashing to parameter ranges.")
    # Neuromodulator-inspired modulation knobs.
    p.add_argument("--nm_enable", type=str2bool, nargs="?", const=True, default=False,
                   help="Enable neuromodulator-based modulation (snn_add/ann_sub/ann_add/ann_combo).")
    p.add_argument("--nm_counts", type=str, default="[0,0]",
                   help="List of neuromodulator type counts per layer (e.g. \"[6,5]\" for hidden/output).")
    p.add_argument("--nm_init_scale", type=float, default=SOFT_MOD_GAIN_INIT,
                   help="Scale for initializing shared neuromodulator MLP weights (small near-zero default).")
    p.add_argument("--nm_debug_print", type=str2bool, nargs="?", const=True, default=False,
                   help="Print NM mapper effect stats at t=0 (debug).")
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
                   choices=["ann_sub", "ann_add", "ann_combo", "snn_add", "snn_sub", "mlp_sub", "mlp_add", "mlp_combo", "all"],
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
                        "alpha_hetero_2,beta_hetero_2,thresholds_1,reset_1,rest_1,input_delay_logits).")

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
        use_input_delay=bool(args.use_input_delay),
        input_delay_steps=max(1, int(args.input_delay_steps)) if args.use_input_delay else 0,
        input_delay_init=args.input_delay_init,
        input_delay_init_cap=int(args.input_delay_init_cap) if args.input_delay_init_cap is not None else None,
        input_delay_init_bias=float(args.input_delay_init_bias),
        input_delay_init_noise=float(args.input_delay_init_noise),
        input_delay_temp=float(args.input_delay_temp),
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
    ann_combo_additive = parse_str_list(args.ann_combo_additive)
    ann_combo_multiplicative = parse_str_list(args.ann_combo_multiplicative)
    ann_combo_additive, ann_combo_multiplicative = _resolve_ann_combo_lists(
        ann_combo_additive, ann_combo_multiplicative
    )

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
        "ann_rnn_state_every_step": bool(args.ann_rnn_state_every_step),
        "mod_update_every_step": bool(args.mod_update_every_step),
        "mod_current_enable": bool(args.mod_current_enable),
        "mod_current_target": str(args.mod_current_target),
        "mod_current_activation": str(args.mod_current_activation),
        "ann_combo_additive": ann_combo_additive,
        "ann_combo_multiplicative": ann_combo_multiplicative,
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
        if normalized_mlp_mode in {"mlp_add", "mlp_combo", "snn_add"}:
            nm_activation_kind = "tanh"
        elif normalized_mlp_mode in {"mlp_sub", "snn_sub"}:
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
    settings["nm_debug_print"] = bool(args.nm_debug_print)
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
            smoothing_modes = ["mlp_sub", "mlp_add", "mlp_combo", "snn_add", "snn_sub"]
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


__all__ = [name for name in globals() if not name.startswith('__')]
