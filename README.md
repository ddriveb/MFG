# MFG-Hedge V2 — October 14 mainline

This directory is the portable extraction of the current October 14 communication-aware allocation study. It combines a four-round FP64 width-16 / price-width-64 GNN, a source-aware five-resource objective and integer realization, the fused CUDA allocator and feature refresh, fixed-50-step Sinkhorn, a Lazarus ring baseline, and separate G64, multiscale, hotspot, and G4 evidence panels.

The older `communication-social-v2-clean` package records the October 11 method. Its README now marks it as historical, with the pre-banner text preserved under `docs/history/`; it should not be read as the October 14 mainline.

## Start with frozen replay

Use Python 3.10 or later with only the standard library to reproduce the frozen plans, quotas, objective values, certificates, and headline summaries. From this directory:

```bash
python -I ./tools/replay_frozen.py --scope all
```

To save a report, choose a new path under `artifacts/`; existing files are not overwritten:

```bash
python -I ./tools/replay_frozen.py --scope all --output artifacts/replay-20261014-a.json
```

The report includes the corrected 150-state G64 comparison, the focal 38-state comparison, all six hotspot specialist cells, and the 150-group G4 saved timing summary. The reported times are frozen measurements. Replaying them checks their records; it does not claim new execution time.

The replay also regenerates each campaign's Gate inputs from the included seeds and protocols, verifies saved plan quotas and objective values, and recomputes the included reference and price certificates. `evidence-manifest.json` covers the packaged campaign records, model checkpoints, licenses and external source snapshots.

## Fresh planner execution

For new Gate draws and GPU inference, use an existing environment with PyTorch, CUDA, and Triton. No environment, package, model, or compiler is downloaded or installed by these commands.

```bash
python -I ./tools/fresh_planners.py --campaign all
```

The run covers all 150 G64, 192 multiscale, and 96 global-hotspot states. It compares GNN, Sinkhorn, and Lazarus integer outputs against the saved records, requires exact quotas, and checks GNN shares and price bounds. Each run writes a new directory below `artifacts/`. Fresh wall-clock times are machine-dependent.

The archived optimized G4 fused ports have their own separate parity entry point. First run the CPU zero-count support/feature characterization; then, on a CUDA/Triton machine, optionally compare the fused ports to the Torch reference on recorded G4 count vectors:

```bash
python -I ./tools/physical_g4_qualify.py --mode zero-template
python -I ./tools/physical_g4_qualify.py --mode recorded-planner --limit-states 1 --output artifacts/physical-g4-parity-smoke-20261014-a
```

The recorded planner check uses one CUDA device and reports fused-kernel parity only. It is not the original four-rank model-training timing test. The parameterized four-rank runner is `tools/run_physical_g4.py`; it supports the archived four-round configuration, defaults to two warmup and 150 measured groups, and has no lease deadline. Its source transformation self-test is safe without GPUs:

```bash
python -I ./tools/run_physical_g4.py --selftest
```

A new measured run requires four visible GPUs, CUDA/Triton, a caller-provided Granite model directory matching `external/granite_model_manifest.json`, and the compiled native schedule library. The launcher checks every pinned model file and size locally before importing Torch, sets the original `CUBLAS_WORKSPACE_CONFIG=:4096:8`, and forces Hugging Face offline mode; it never downloads missing files. From the WSL/Linux CUDA environment with `torchrun`, run:

```bash
export PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
torchrun --standalone --nnodes=1 --nproc-per-node=4 ./tools/run_physical_g4.py --config ./configs/g4-optimized-paired.json --output artifacts/physical-g4-full-20261014-a --model-dir /models/granite-3.0-1b-a400m-instruct --schedule-library /native/granite-schedule.so
```

You may pass `--deadline` with a new future ISO-8601 timestamp that includes a timezone. No historical lease deadline is inherited. The runner keeps the old qualification's strict full-model numerical failure visible in its output. The unmodified historical `optimized_full130.py` and `qualify_optimized130.py` remain under `external/physical_g4/optimized130/` for provenance; use `tools/run_physical_g4.py` for a fresh launch. `external/replay-entrypoints.json` records optional backend requirements.

The fused four-rank launcher rejects the archived runner's `--local-tiny`/`--init-file` Gloo test mode: its optimized ports require CUDA. Use `--selftest` for its CPU-only launch checks. A config with `rounds` other than 4 is rejected before GPU runtime setup.

## Offline training

The portable training entry points preserve each source protocol's default schedule, model, optimizer, seed, and checkpoint selection. The full defaults are 4,000 allocation steps, 2,000 price steps, 256 train and 64 dev cases per scale, with batch size 2 per scale. Training writes only to a new `artifacts/` run directory and does not modify the packaged initial checkpoint.

Use explicit short budgets for a smoke run:

```bash
python -I ./tools/train_multiscale.py --allocation-steps 2 --price-steps 2 --eval-every 1 --train-per-scale 2 --dev-per-scale 2 --batch-per-scale 2 --output artifacts/training-multiscale-smoke-20261014-a
python -I ./tools/train_global_hotspot.py --allocation-steps 2 --price-steps 2 --eval-every 1 --train-per-scale 2 --dev-per-scale 2 --batch-per-scale 2 --output artifacts/training-hotspot-smoke-20261014-a
```

Optional arguments include `--device`, `--initial-checkpoint`, `--python-include`, and `--triton-cache`. Initial checkpoints must be inside `models/checkpoints/`; build and cache paths are caller-provided. Reduced training runs are labelled `smoke`, and other changes from the frozen defaults are labelled `custom`. Retraining across devices is not promised to produce bitwise-identical weights.

## External reproductions

- **Granite full-step experiment:** The pinned model revision and expected file manifest are in `external/granite_model_manifest.json`. Weights are not included. A new four-rank run additionally needs a local model directory and a locally built native scheduling library. Use `tools/run_physical_g4.py` with `configs/g4-optimized-paired.json`; saved results are replayable without either asset.
- **G4 fused planner ports:** The archive records the optimized CUDA/Triton allocation, Sinkhorn, and Lazarus ports used in the physical G4 study. Its saved 1,694-case qualification retained the strict full-model numerical failure (`strict_fullmodel_numerics_passed: false`) alongside passing quota/share checks. The new `physical_g4_qualify.py` reruns planner parity independently and does not convert that historical strict-fail into a pass.
- **TAOT comparison:** `external/taot/` contains the pinned source excerpt, upstream license, and source manifest. The V2 fixed-50 Sinkhorn planner runs without importing TAOT.
- **ns-3 matrix replay:** `external/ns3/MfgMatrixReplay.cc` is the custom source excerpt. Building it requires the separately pinned ns-3 checkout/toolchain; see the command and argument order in `external/replay-entrypoints.json`.

All evidence paths resolve beneath this directory. The lightweight `src/mfg_hedge/__init__.py` intentionally avoids importing the original workspace's V1 exports or its editable environment. Optional Python modules loaded by a V2 command must resolve under `v2/src/mfg_hedge`. V2 commands perform no implicit download or dependency installation; optional external executions use only caller-supplied local assets.
