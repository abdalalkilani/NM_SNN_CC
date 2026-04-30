# SNN all-in-one runner

Fresh split of `../snn_allinone.py` into smaller modules. The code is the same runner, just easier to browse.

## Run It

From inside this folder:

```bash
cd path/to/this_folder
python __main__.py --help
```

From the parent folder:

```bash
cd path/to/parent_folder
python -m this_folder_name --help
```

`python -m this_folder_name` is the normal Python package way to run it. It makes Python load the folder as a package, so imports like `from .cli import main` work correctly. If you are already inside the folder, use `python __main__.py` instead.

Basic runs:

```bash
# Train only the base SNN
python __main__.py --run_mode snn

# Train modulated SNN directly
python __main__.py --run_mode mod

# Train SNN first, then train modulator from the chosen SNN checkpoint
python __main__.py --run_mode staged
```

Use `--flag true` / `--flag false` for boolean options. Most bool flags also work as a bare flag, for example `--train_aug_enable`.


## File Map

- `__main__.py`: small entry point. Lets you run `python __main__.py` from inside this folder, or `python -m package_name` from the parent folder.
- `cli.py`: all command-line options and the main run flow for `snn`, `mod`, and `staged`.
- `config.py`: imports, device/dtype setup, constants, parameter names, and default ranges.
- `utils.py`: small shared helpers for parsing, run folders, train flags, clamping, tau helpers, and parameter maps.
- `data.py`: H5 loading, validation loading, spike augmentation, channel compression helpers, batching, and split saving/loading.
- `snn.py`: base SNN spike function, input delay, model setup, forward pass, eval, test, and SNN training loop.
- `modulation.py`: ANN/RNN/LSTM modulators, SNN add/sub modulators, modulator init, IO slices, and `build_modulator`.
- `grouping.py`: grouped/overlap modulation, neuromodulator mapper, fixed masks, param timescales, and smoothing.
- `modulated.py`: modulated checkpoint loading, modulated forward pass, mod eval/test, spike stats, SHAP analysis, and modulated training loop.
- `reporting.py`: parameter count summaries, shape summaries, run config JSON, and readable training log header.
- `__init__.py`: public imports for common functions/classes.

## Common Examples

Train a base SNN with custom size:

```bash
python __main__.py \
  --run_mode snn \
  --nb_inputs 700 \
  --nb_hidden 256 \
  --nb_outputs 20 \
  --nb_steps 100 \
  --nb_epochs 30
```

Train a modulated model from an existing SNN checkpoint:

```bash
python __main__.py \
  --run_mode mod \
  --base_snn_ckpt Runs_SNN_EXP/Run_1/Fold_1/snn_best.pth \
  --ann_mode ann_sub \
  --ann_hidden_sizes "[2048]" \
  --ann_interval 3
```

Use SNN modulation instead of ANN modulation:

```bash
python __main__.py \
  --run_mode mod \
  --ann_mode snn_add \
  --mod_hidden_sizes "[512]" \
  --snn_mod_gain_init 0.01
```

Enable grouping/overlap:

```bash
python __main__.py \
  --run_mode mod \
  --group_size 5 1 \
  --group_overlap 2 0 \
  --ann_mode ann_sub
```

Enable neuromodulator style mapping:

```bash
python __main__.py \
  --run_mode mod \
  --nm_enable true \
  --nm_counts "[5,2]" \
  --nm_warm_start true
```

## Main Options

Core model/data:

- `--run_mode`: `snn`, `mod`, or `staged`.
- `--nb_inputs`, `--nb_hidden`, `--nb_outputs`: model sizes.
- `--time_step`, `--nb_steps`, `--max_time`: simulation timing.
- `--batch_size`: minibatch size.
- `--cache_dir`, `--cache_subdir`, `--train_file`, `--test_file`, `--val_file`: H5 data locations.

Training:

- `--lr`: learning rate.
- `--nb_epochs`: epochs for single phase runs.
- `--nb_epochs_snn`, `--nb_epochs_mod`: staged SNN/modulator epochs.
- `--tau_syn`, `--tau_mem`: SNN time constants.
- `--tau_match_clip`: use `max_time / nb_steps` as decay step.
- `--weight_scale`: base weight scale.
- `--psp_norm_peak`: normalize PSP peaks.
- `--homo_init`: use homogeneous SNN init instead of hetero init.
- `--save_every_epoch`: save every epoch checkpoint.

Save dirs:

- `--save_dir_root`: shared root. Creates `SNN/` and `NM/` under it.
- `--save_dir_snn`: SNN save folder.
- `--save_dir_mod`: modulated save folder.

Regularization:

- `--snn_reg_enable`, `--mod_reg_enable`, `--snn_mod_reg_enable`: enable spike regularizers.
- `--snn_reg_scale`, `--mod_reg_scale`, `--snn_mod_reg_scale`: regularizer weights.

Input delay:

- `--use_input_delay`: enable learnable per-input delays.
- `--input_delay_steps`: max delay steps.
- `--input_delay_init`: fixed initial delay, or random if omitted.
- `--input_delay_init_cap`: cap random delay init.
- `--input_delay_init_bias`, `--input_delay_init_noise`: init strength/noise.
- `--input_delay_temp`: softmax temperature for delay choice.

Augmentation/noise:

- `--train_aug_enable`, `--eval_aug_enable`, `--test_aug_enable`: enable augmentation per split.
- `--aug_jitter_ms`, `--aug_shift_ms`, `--aug_scale_low`, `--aug_scale_high`: time transforms.
- `--aug_drop_p`, `--aug_insert_rate`, `--aug_band_frac`: event-level transforms.
- `--aug_channel_jitter_std`: channel jitter.
- `--aug_noise_rate_hz`, `--aug_noise_per_input`: Poisson noise.
- `--train_noise_enable`, `--eval_noise_enable`, `--test_noise_enable`: enable noise per split.
- `--paper_aug_train`, `--paper_aug_eval`, `--paper_aug_test`: use paper-style aug/noise only.
- `--paper_tau_scale_enable`, `--paper_tau_scale`: scale tau values.
- `--postbin_time_mask_train`, `--postbin_time_mask_eval`, `--postbin_time_mask_test`: mask binned time windows.
- `--hidden_dropout_p`: hidden spike dropout during training.

Channel compression:

- `--channel_compress_enable`: enable input channel compression.
- `--channel_compress_target`: target compressed channel count.
- `--channel_compress_mode`: `all`, `mod_only`, or `mod_mlp`.
- `--channel_compress_mlp_hidden_sizes`: hidden sizes for learned modulator input compression.

Validation/splits:

- `--use_validation`: use validation.
- `--val_fraction`: train/val split fraction.
- `--k_folds`: number of folds.
- `--patience`: early stopping patience.
- `--seed`: random seed.
- `--fixed_split_path`: load saved split indices.
- `--reinit_per_fold`: reinitialize model per fold.

Staged/unlocking:

- `--staged_unlock_metric`: `train_acc` or `val_acc`.
- `--staged_unlock_threshold`: threshold for staged chosen checkpoint.
- `--staged_min_epochs`: reserved/min epochs setting.
- `--start_locked`: freeze SNN params at start of mod training.
- `--unlock_metric`: `train_acc` or `val_acc`.
- `--unlock_threshold`: metric threshold to unlock SNN params.

ANN/SNN modulator:

- `--ann_mode`: `ann_sub`, `ann_add`, `ann_combo`, `snn_add`, or `snn_sub`.
- `--ann_hidden_sizes`: ANN/RNN/LSTM hidden sizes, e.g. `"[2048,1024]"`.
- `--ann_interval`: modulation update interval.
- `--ann_arch`: `mlp`, `rnn`, or `lstm`.
- `--ann_rnn_state_every_step`: advance RNN/LSTM state every timestep.
- `--mod_update_every_step`: apply latest modulation every timestep.
- `--ann_combo_additive`, `--ann_combo_multiplicative`: params for combo mode.
- `--ann_in_disable`: comma-separated input blocks to remove.
- `--ann_out_disable`: comma-separated output blocks to remove.
- `--ann_output_activation`: `linear`, `sigmoid`, `tanh`, `default`, or `none`.
- `--mod_current_enable`, `--mod_current_target`, `--mod_current_activation`: use modulator output as injected current.
- `--mod_hidden_sizes`: hidden sizes for SNN modulator.
- `--snn_mod_hidden`: deprecated, kept for compatibility.
- `--snn_mod_hidden_recurrent`, `--snn_mod_rec_init_zero`: recurrent SNN modulator settings.
- `--snn_mod_gain_init`, `--snn_mod_weight_scale`: SNN modulator init.
- `--snn_add_balanced_init`, `--snn_add_init_effect_frac`: balanced add init.
- `--snn_sub_scale_init`, `--snn_sub_bias_init`: substitution init.

Neuromodulator features:

- `--nm_enable`: enable NM mapping.
- `--nm_counts`: hidden/output NM counts, e.g. `"[5,2]"`.
- `--nm_init_scale`: NM mapper init scale.
- `--nm_debug_print`: print NM debug stats.
- `--nm_neuron_frac_enable`, `--nm_neuron_frac`: fraction of neurons modulated.
- `--nm_param_frac_enable`, `--nm_param_frac`: fraction per parameter.
- `--nm_mapper_type`: `mlp` or `linear`.
- `--nm_mapper_activation`: `auto`, `linear`, `sigmoid`, `tanh`, or `none`.
- `--nm_mapper_hidden_activation`: `silu`, `gelu`, `leakyrelu`, `tanh`, `relu`, or `none`.
- `--nm_mapper_hidden_size`: override mapper hidden width.
- `--nm_flat_order`: `type_major` or `target_major`.
- `--nm_warm_start`, `--nm_warm_start_force_linear`: warm-start NM as close to normal mode as possible.

Fixed masks/grouped flat inputs:

- `--mod_fixed_mask_enable`: use one fixed modulation mask for the run.
- `--mod_fixed_mask_seed`: seed for mask sampling.
- `--mod_fixed_mask_flat_inputs`: also shrink `hid_flat`/`out_flat`.
- `--mod_fixed_mask_zero_fallback`: old fallback behavior for 0 fractions.
- `--mod_hid_flat_group`: use grouped hidden flat inputs.
- `--mod_hid_flat_modulated_only`: keep only modulated hidden flat inputs.

Grouping:

- `--group_size`: hidden/output group size, e.g. `5 1`.
- `--group_overlap`: hidden/output overlap, e.g. `2 0`.
- `--group_distribution`: `uniform` or `normal` per layer.
- `--group_normal_std`: normal group std per layer.

Timescales/smoothing:

- `--param_timescales_enable`: learn/update per-param intervals.
- `--param_timescales_distribution`: `fixed`, `uniform`, or `normal`.
- `--param_timescales_scale`, `--param_timescales_std`, `--param_timescales_seed`: interval sampling.
- `--param_timescales_trainable`: make intervals trainable.
- `--param_smoothing_enable`: smooth parameter changes over time.
- `--param_smoothing_tau_init`, `--param_smoothing_tau_min`: smoothing mix values.
- `--param_smoothing_trainable`: make smoothing trainable.
- `--param_smoothing_modes`: modes to smooth, or `all`.

Parameter ranges:

- `--alpha_1_range MIN MAX`
- `--beta_1_range MIN MAX`
- `--thr_range MIN MAX`
- `--reset_range MIN MAX`
- `--rest_range MIN MAX`
- `--alpha_2_range MIN MAX`
- `--beta_2_range MIN MAX`

Analysis:

- `--ann_shap_enable`: run approximate Shapley analysis after mod/staged.
- `--ann_shap_samples`: number of permutations.
- `--ann_shap_batch_limit`: batches to cache.
- `--ann_shap_dataset`: `train` or `test`.
- `--ann_shap_metric`: `acc` or `nll`.

Base checkpoint options:

- `--base_snn_ckpt`: path to a base SNN checkpoint for `--run_mode mod`.
- `--base_snn_from_stockpile`: pick a random base SNN from a stockpile.
- `--base_snn_stockpile_dir`: stockpile directory.
- `--base_snn_stockpile_seed`: optional seed for stockpile selection.

SNN train freezing:

- `--snn_train_disable`: comma-separated SNN params to freeze, e.g. `w1,w2,v1,reset_1`.

Valid SNN param names are `w1`, `w2`, `v1`, `alpha_hetero_1`, `beta_hetero_1`, `alpha_hetero_2`, `beta_hetero_2`, `thresholds_1`, `reset_1`, `rest_1`, and `input_delay_logits`.

## Python Use

Import from the package when using it in another script:

```python
from package_folder_name import main
from package_folder_name import setup_model, train_snn_hetero
from package_folder_name import build_modulator, train_modulated_snn

main(["--run_mode", "snn", "--nb_epochs", "1"])
```

The most useful public functions are:

- `main(argv=None)`: command-line entry point.
- `setup_model(settings)`: build base SNN parameter state.
- `run_snn_hetero(inputs_dense, state, settings)`: base SNN forward.
- `train_snn_hetero(...)`: train base SNN.
- `build_modulator(settings, override_mode=None, hidden_sizes=None)`: build ANN/SNN modulator.
- `run_snn_modulated(inputs, state, settings, modulator, mlp_interval, ...)`: modulated forward.
- `train_modulated_snn(...)`: train modulated SNN.
- `load_base_snn_state(...)`, `load_modulated_checkpoint(...)`: checkpoint loading.
