# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

FedCircuitNet is the **implementation** of the thesis project: a federated
learning pipeline for DRC violation prediction on CircuitNet-N28, where FL
clients simulate EDA design teams and the research contribution is the
**data partitioning scheme**.

`../../CLAUDE.md` (repo root, `fl_chip_thesis/`) holds the research
methodology — dataset properties, filename conventions, analysis phases and
pre-registered decision thresholds. **Read it for "why"; read this file for
"how the code works".** Do not duplicate methodology here.

## Commands

```bash
# Environment (Python 3.11)
python3.11 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt

# Build metadata CSVs: translates upstream CircuitNet two-column ann-files
# into the filename+factor schema this repo needs. N28/N14 schema is
# auto-detected per file.
./build_all_metadata.sh
python create_metadata_files.py --input=<ann.csv> --output=files/<name>.csv

# Train / test one scheme (Hydra picks the config by name, no .yaml suffix)
python train_aim.py --config-name=fedavg_train_feature_hierarchical
python test_aim.py  --config-name=fedavg_test_feature_hierarchical

# All scheme pairs sequentially (iid, kmeans, dirichlet, feature_hierarchical)
./run_all_experiments.sh

# Any nested config field is overridable from the CLI
python train_aim.py --config-name=fedavg_train_iid \
    training.num_rounds=2 strategy.local_epochs=2 runtime.cpu=true \
    partitioning.n_partitions=4 data.metadata_csv=./files/train_N28.csv
```

There is **no test suite, linter or build step**. Verify changes by running a
short training job with `training.num_rounds` / `strategy.local_epochs`
overridden, or by exercising a partitioner directly against
`files/train_N28.csv` (7,077 rows) and asserting the partition is complete and
disjoint. `partitioning_analysis.ipynb` compares partitioners; `show_result.ipynb`
reads training/test results back out of the local Aim repo.

## Architecture

**Everything is config-driven through four independent registries.** Adding a
variant never requires touching `train_aim.py` / `test_aim.py` — implement the
class, register it, then name it in a YAML `type:` field.

| Registry | Location | Selected by |
|---|---|---|
| `PARTITIONER_REGISTRY` | `partitioning/__init__.py` | `partitioning.type` |
| `PREPROCESSOR_REGISTRY` | `partitioning/preprocessing.py` | `partitioning.preprocessing[].type` |
| `MODEL_REGISTRY` | `models/__init__.py` | `model.type` |
| `STRATEGY_REGISTRY` | `federated/strategies/__init__.py` | `strategy.type` |

Each registry has a `_build_*` / `build_*` helper that pops `type` and forwards
the remaining config keys as kwargs. Partitioners accept `**kwargs` so extra
YAML keys are tolerated.

### FL pipeline (template method)

`FederatedStrategy.train()` (`federated/strategy.py`) owns the entire pipeline:
partition metadata → build a `DataLoader` per partition → build clients → run
synchronous rounds → collect `RoundStats`. Concrete strategies implement only
two hooks: `_build_client()` and `_build_aggregator()`. FedAvg lives in
`federated/strategies/fedavg/` (client = local SGD, aggregator = sample-weighted
average). A new algorithm (FedProx, SCAFFOLD, FedBN) is a new subpackage +
registry entry, nothing more.

`FederatedServer` (`federated/server.py`) runs clients through a
`ThreadPoolExecutor`, one CUDA stream per client so their GPU work overlaps.

### Partitioning (the project's core contribution)

Two families under `partitioning/`:

- **Direct** — `IIDPartitioner`, `DirichletPartitioner` implement
  `DatasetPartitioner.partition()` themselves.
- **FedChip-style** — `FedChipPartitioner` (`partitioning/base.py`) implements
  `partition()` as a shared two-stage template: subclasses supply
  `_assign_groups(df) -> group_ids`, then the base applies the FedChip
  ownership + spillover stage (each client keeps `cluster_share`, default 80 %,
  of its group; the rest is redistributed across all clients via a fresh
  `Dirichlet(dirichlet_alpha)` draw). Subclasses:
  - `KMeansClustering` — groups from QuantileTransformer → StandardScaler →
    PCA → k-means on the metadata matrix.
  - `FeatureHierarchicalPartitioner` — groups from a recursive top-down split
    on an **ordered list of raw metadata columns** given in config. No
    preprocessing by design: there is no distance metric here, only exact
    grouping, and this keeps clients describable in EDA terms and avoids the
    circularity of feeding cluster labels back in as client labels.

All partitioners expose `stratify_cols`, which `train_aim.py` uses to log
per-partition factor histograms to Aim at round 0.

**Preprocessing (`partitioning/preprocessing.py`) is invisible downstream by
design.** `DatasetPartitioner._views(df)` returns a `(work, out)` pair: every
transform in the `partitioning.preprocessing` block is applied to `work`, which
is what the partitioner groups on, but only transforms flagged
`keep_in_output: true` are applied to `out`, which is what the returned
partitions are sliced from. So a client can be *formed* on a coarsened factor
while every row it hands back — and therefore the Aim histograms, the JS /
coverage tables in `partitioning_analysis.ipynb`, and `DRCDataset` — still
carries the raw metadata value. Both frames keep the input row order, so the
positional assignment computed on `work` indexes `out` unchanged; transforms
must be row-preserving for this to hold.

`rank_utilization` is the one shipped transform (`PARTITIONING_ANALYSIS.md`
§4.1): utilization's five N28 levels collapse to `low` (0.70, 0.75) / `medium`
(0.80) / `high` (0.85, 0.90), cutting the design × utilization grid from 30
cells to 18. It is enabled only in `fedavg_train_feature_hierarchical.yaml`;
the iid / dirichlet / kmeans configs carry it commented out so those baselines
stay comparable to already-run results. The ranked column becomes an *ordered*
`Categorical`, which two consumers depend on: `_split` groups with `sort=True,
observed=True` so ranks sort low→high (not alphabetically) and empty ranks
don't consume split budget, and `_prepare_matrix` maps ordered categoricals to
integer codes instead of one-hot so a ranked factor keeps the single-column
weight the raw numeric factor had.

### Data flow

`files/*.csv` (one row per sample: `filename` + design factors) → partitioner
→ per-client DataFrame → `DRCDataset` (`datasets/drc_dataset.py`) loads
`{feature_dir}/{filename}.npy` (9×H×W) and `{label_dir}/{filename}.npy`.
`filename` is the join key throughout and is required by `_validate()`.

### Config contract

Train configs carry `data / partitioning / model / strategy / runtime /
training / evaluation`. **Test configs have no `partitioning` block** —
`test_aim.py` evaluates the saved global checkpoint centrally over the whole
test CSV, then runs the multi-threshold ROC/PR-AUC sweep from `utils/metrics.py`.
Config pairs must agree: `training.save_path` (train) ==
`checkpoint`'s parent dir and `evaluation.save_path` (test).

## Gotchas

- **`strategy.local_epochs` is a step cap, not epochs.** `FedAvgClient.local_update`
  iterates the loader once and breaks after `local_epochs` minibatches. The LR
  schedule (`CosineRestartLr`) restarts each round over that step budget.
- **`build_loss()` mutates its argument** — it `pop`s `loss_type` off the dict.
  Calling it twice on the same config block raises `KeyError`.
- **In-training validation is currently broken.** `train_aim.py:195` passes
  `threshold` (a float) into the `strategy_cfg` parameter of
  `_evaluate_global_model`, which then does `strategy_cfg['eval_metric']`. All
  shipped configs set `evaluation.metadata_csv: null`, so it never fires. Fix
  the call site before enabling per-round eval.
- **No client sampling.** Every client trains every round; there is no
  fraction-`C` selection despite the FedAvg pseudo-code in the docstrings.
- **Configs hardcode absolute `/home/spedicato/...` feature/label paths** for the
  remote GPU box. Override `data.feature_dir` / `data.label_dir` when running elsewhere.
- **`fedavg_test_iid.yaml` and `fedavg_test_kmeans.yaml` share
  `experiment: fedavg_drc_test`.** `show_result.ipynb` disambiguates runs by
  checkpoint directory, not experiment name.
- **`utils/` and `models/routenet.py` are verbatim ports** from
  `code_examples/CircuitNet/drc_prediction`, kept identical so federated results
  stay comparable to the centralised RouteNet baseline (ROC-AUC 0.95 / PR-AUC 0.63).
  Do not "improve" them; changes must be mirrored in both places.
- `build_metric(name)` lowercases and looks up a module global, so config
  `NRMS` / `SSIM` map to `nrms` / `ssim`. Available: `nrms`, `ssim`, `psnr`, `emd`.
- **Always report PR-AUC alongside ROC-AUC** — the DRC labels are heavily
  imbalanced and ROC-AUC alone overstates performance.
