# Reproducibility Guide

## What This Repository Reproduces

The code supports the experiment protocols reported in the manuscript:

1. matched selective-risk comparison across completed DTA backbones;
2. thresholded risk-control evaluation;
3. named strong-backbone posthoc evaluation;
4. rolling ChEMBL release-temporal backtesting;
5. decision-budget virtual screening;
6. failure-mode decomposition;
7. block-bootstrap, leave-one-group-out, and independent-calibration audits.

The repository does not include generated figures, embedded manuscript tables, large prediction files, trained checkpoints, or full result snapshots.

## Expected Directory Convention

Run scripts from the repository root. By default, data loaders expect prepared benchmark inputs below `data/processed/`, and generated outputs are written beneath `reports/` or related run-specific output directories. Both paths are ignored by Git.

For the rolling release audit, the required input is:

```text
data/processed/chembl/standardized_pairs.csv
```

The file must include the pair identifiers, affinity target, molecular/protein fields required by the configured backbones, and a `chembl_release` column.

## Core Commands

```bash
pip install -e .
python scripts/check_env.py
python scripts/run_trans_grade_experiments.py
python scripts/run_maximal_trans_experiments.py
python scripts/run_chembl_rolling_release_audit.py --workspace .
python scripts/run_submission_upgrade_audits.py --workspace .
```

## Audit Outputs

`run_submission_upgrade_audits.py` generates:

- named strong-backbone paired comparisons;
- hierarchical/block bootstrap summaries;
- leave-one-group-out robustness summaries;
- independent-calibration excessive-error risk-limit summaries.

`run_chembl_rolling_release_audit.py` generates:

- rolling-window design records;
- confidence-source summary metrics;
- paired selector-versus-comparator summaries;
- optional pair-level prediction outputs.

These generated files are excluded from this repository to keep the public release compact.

## Calibration Assumptions

The independent-calibration risk-limit audit uses separate selector-fitting and calibration subsets together with simultaneous one-sided binomial upper bounds. Its finite-sample interpretation requires the stated calibration assumptions and does not imply validity under arbitrary distribution shift.

