# RCSDTA

Reference implementation for the manuscript **Selective Reliability for Drug--Target Affinity Prediction under Distribution Shift**.

RCSDTA attaches a posthoc residual-risk selector to fixed drug--target affinity (DTA) predictors. It supports selective retention, independent-calibration risk-limit audits, ChEMBL temporal evaluation, decision-budget virtual screening, and failure-mode analysis.

## Release Scope

This repository is intentionally lightweight and code-focused. It includes:

- core implementation under `src/selective_dta_b/`;
- experiment launch and audit scripts under `scripts/`;
- split and experiment configurations under `configs/`;
- Python dependency specifications.

Generated predictions, trained checkpoints, figures, manuscript tables, and large result snapshots are excluded. They can be recreated by running the documented pipelines after obtaining the public benchmark data and required pretrained assets.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-torch.txt
pip install torch-geometric
pip install -e .
```

On Windows PowerShell, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Main Reproduction Workflows

Start with environment validation:

```bash
python scripts/check_env.py
python scripts/detect_resources.py
```

Run the locked primary-selector workflow used for the submission evidence chain:

```bash
python scripts/run_primary_submission_experiments.py --workspace . --output-dir reports/primary_submission_experiments
```

The primary selector is fixed before test evaluation as `Ridge(alpha=1.0)` on the `enriched9` residual-risk feature set. The decision-budget protocol uses a fixed risk-adjusted candidate score,
`prediction_mean - 1.0 * predicted_abs_error`, and records how often its recommendations differ from prediction-only ranking.

Run the expanded ChEMBL publication-year temporal backtest:

```bash
python scripts/run_chembl_temporal_backtest.py \
  --workspace . \
  --output-dir reports/primary_submission_experiments/chembl_expanded \
  --refresh \
  --train-max-rows 9000 \
  --val-max-rows 4500 \
  --test-max-rows 6000
python scripts/run_chembl_rolling_release_audit.py \
  --workspace . \
  --output-dir reports/primary_submission_experiments/chembl_expanded_rolling
```

The publication-year command obtains public ChEMBL records through the API and records its requested sampling caps in the output status file. Sampling-scale-specific caches prevent an expanded run from being silently confused with the smaller default protocol. The rolling ChEMBL audit uses the materialized `chembl_release` metadata to diagnose release-to-release temporal transfer.

Additional legacy and sensitivity workflows remain available:

```bash
python scripts/run_trans_grade_experiments.py
python scripts/run_maximal_trans_experiments.py
python scripts/run_submission_upgrade_audits.py --workspace .
```

## Backbones and Evaluation

The provided code covers the posthoc selector pipeline and experiment adapters used for classical, neural, graph, transformer, and externally produced DTA outputs. External backbone predictions must be supplied in the expected pair-level schema when model weights or third-party repositories are not distributed here.

The paper's main evidence blocks are:

- paired selective reliability and retained-set error;
- independent-calibration excessive-error risk-limit audits;
- named strong-backbone posthoc transfer;
- rolling ChEMBL release-temporal backtests;
- decision-budget virtual screening with novel-target subgroup summaries;
- negative/failure-mode analysis.

## Data and Large Assets

Benchmark data are not redistributed in this repository. Prepare Davis, KIBA, and BindingDB data according to their respective licenses and the loaders/configurations in this package. The ChEMBL publication-year protocol can materialize its public data directly through `run_chembl_temporal_backtest.py`; the generated records and caches remain excluded. Large pretrained molecular/protein assets and output predictions are also excluded.

## Claim Boundary

RCSDTA is a retrospective reliability and risk-aware triage framework. This release does not claim prospective wet-lab validation, universal temporal AURC gains, or a distribution-free guarantee under arbitrary data shifts. Risk-limit results have their finite-sample interpretation only under the documented independent-calibration assumptions.

## License

See [LICENSE](LICENSE).
