# Small SNN All-In-One

This is the smaller reader-friendly version of the SNN / modulated SNN runner.

It is not as aggressively modular as `../`, but it is easier to read than the original giant `snn_allinone.py`.

## Files

- `snn_allinone_small.py`: model/data/forward code.
- `snn_allinone_small_cli.py`: training, evaluation, checkpoints, logging, analysis, and command-line options.

## What Is Kept

- Base SNN model and forward pass.
- H5 data loading, batching, splits, and augmentation helpers.
- Channel compression.
- ANN modulation with MLP `ann_sub` and `ann_add`.
- SNN additive modulator with `snn_add`.
- Modulated forward pass and staged training support.
- Neuromodulator mapping, without debug-print mode.
- Grouping, overlap, fixed masks, parameter ranges, timescales, and smoothing.

## What Is Removed

- Input delay support.
- `ann_combo`.
- RNN/LSTM modulators.
- `snn_sub`.
- Neuromodulator debug printing.

## Run It

From this folder:

```bash
python snn_allinone_small_cli.py --help
```

Train only the base SNN:

```bash
python snn_allinone_small_cli.py --run_mode snn
```

Train a modulated SNN directly:

```bash
python snn_allinone_small_cli.py --run_mode mod
```

Train base SNN first, then train the modulator:

```bash
python snn_allinone_small_cli.py --run_mode staged
```

## Reading Order

Start with `snn_allinone_small.py`:

1. Constants and helper functions.
2. Data loading and augmentation.
3. Base SNN model and forward pass.
4. ANN/SNN modulator classes.
5. Modulated SNN forward pass.

Then open `snn_allinone_small_cli.py` if you want to see how training, checkpoints, and command-line arguments are wired together.
