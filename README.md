# ALL_IN_ONE

This folder contains the current SNN / modulated SNN runner code.

## Main Code

- `snn_allinone.py`: original all-in-one research script. Full functionality, but large.
- `snn_allinone_split/`: cleaned GitHub-facing version with clearer structure.
- `snn_allinone_split/small/`: smaller two-file version for readers who mainly want to understand the model and core ideas.

## Which Version To Use

Use `snn_allinone_split/` if you want the full package-style version with separate modules.

Use `snn_allinone_split/small/` if you want a simpler version:

- model/data/forward code in one file
- CLI/training/checkpoint code in another file
- some advanced options removed to make it easier to read

The old `snn_allinone.py` is kept as the original reference.

## Quick Start

Full package:

```bash
cd snn_allinone_split
python __main__.py --help
```

Small version:

```bash
cd snn_allinone_split/small
python snn_allinone_small_cli.py --help
```

Both versions expect the same main Python dependencies used by the original script, including PyTorch, NumPy, and h5py.
