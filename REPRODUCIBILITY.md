# Reproducibility Guide

## What This Repository Reproduces

The code supports the experiment protocols reported in the manuscript:

1. matched selective-risk comparison across completed DTA backbones;
2. independent-calibration excessive-error risk-limit evaluation;
3. named strong-backbone posthoc evaluation;
4. rolling ChEMBL release-temporal backtesting;
5. decision-budget virtual screening;
6. failure-mode decomposition;
7. block-bootstrap, leave-one-group-out, and independent-calibration audits.

The repository does not include generated figures, embedded manuscript tables, large prediction files, trained checkpoints, or full result snapshots.

## Expected Directory Convention

Run scripts from the repository root. By default, data loaders expect prepared benchmark inputs below `data/processed/`, and generated outputs are written beneath `reports/` or related run-specific output directories. Both paths are ignored by Git.

For fixed-output primary evaluation, each validation/test prediction pair must expose:

```text
row_id, dataset_name, target_id, target, prediction_mean,
prediction_std_mc_dropout, target_familiarity, target_novelty
```

Prediction files are discovered beneath `artifacts/` using the repository's existing posthoc-selector directory convention. For the rolling release audit, the required materialized input is:

```text
data/processed/chembl/standardized_pairs.csv
```

The file must include the pair identifiers, affinity target, molecular/protein fields required by the configured backbones, and a `chembl_release` column. It can be generated from public ChEMBL API records by the command below.

## Core Commands

```bash
pip install -e .
python scripts/check_env.py
python scripts/run_primary_submission_experiments.py --workspace . --output-dir reports/primary_submission_experiments
python scripts/run_chembl_temporal_backtest.py --workspace . --output-dir reports/primary_submission_experiments/chembl_expanded --refresh --train-max-rows 9000 --val-max-rows 4500 --test-max-rows 6000
python scripts/run_chembl_rolling_release_audit.py --workspace . --output-dir reports/primary_submission_experiments/chembl_expanded_rolling
```

## Audit Outputs

`run_primary_submission_experiments.py` fixes the submission selector to `Ridge(alpha=1.0)` with the `enriched9` feature set, then generates:

- main and named strong-backbone paired comparisons;
- block-bootstrap summaries;
- independent-calibration excessive-error risk-limit summaries;
- fixed-lambda decision-budget screening summaries, including recommendation-change rates.

`run_chembl_temporal_backtest.py` fetches ChEMBL records with public activity, target, and document metadata; it retains publication year and ChEMBL release fields, and writes the requested train/validation/test caps into `chembl_release_backtest_status.json`. The expanded command above tests train-old/test-new transfer with materially larger acquisition limits than the original default run.

`run_chembl_rolling_release_audit.py` generates:

- rolling-window design records;
- confidence-source summary metrics;
- paired selector-versus-comparator summaries;
- optional pair-level prediction outputs.

These generated files are excluded from this repository to keep the public release compact. They are generated artefacts, not input data required to inspect the protocol.

## Calibration Assumptions

The independent-calibration risk-limit audit uses separate selector-fitting and calibration subsets together with simultaneous one-sided binomial upper bounds. Its finite-sample interpretation requires the stated calibration assumptions and does not imply validity under arbitrary distribution shift. The ChEMBL temporal experiment is therefore reported as a retrospective stress test rather than as a prospective guarantee.
