# RCSDTA

Reference implementation for the manuscript **Risk-Controlled Selective Drug--Target Affinity Prediction under Distribution Shift**.

RCSDTA attaches a posthoc residual-risk selector to fixed drug--target affinity (DTA) predictors. It supports selective retention, risk-control audits, rolling ChEMBL release-temporal evaluation, decision-budget virtual screening, and failure-mode analysis.

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

Run the main selective reliability workflow:

```bash
python scripts/run_trans_grade_experiments.py
python scripts/build_paper_selective_summary.py
```

Run maximal reliability and temporal analyses:

```bash
python scripts/run_maximal_trans_experiments.py
python scripts/run_chembl_temporal_backtest.py
python scripts/run_chembl_rolling_release_audit.py --workspace .
python scripts/run_submission_upgrade_audits.py --workspace .
```

The rolling ChEMBL audit produces the train-old/calibrate/test-new evaluation windows used to diagnose temporal transfer. The submission-upgrade audit computes named strong-backbone comparisons, block-bootstrap robustness, leave-one-group-out analyses, and independent-calibration excessive-error risk-limit summaries.

## Backbones and Evaluation

The provided code covers the posthoc selector pipeline and experiment adapters used for classical, neural, graph, transformer, and externally produced DTA outputs. External backbone predictions must be supplied in the expected pair-level schema when model weights or third-party repositories are not distributed here.

The paper's main evidence blocks are:

- paired selective reliability and retained-set error;
- formal and independent-calibration risk-control audits;
- named strong-backbone posthoc transfer;
- rolling ChEMBL release-temporal backtests;
- decision-budget virtual screening;
- negative/failure-mode analysis.

## Data and Large Assets

Benchmark data are not redistributed in this repository. Prepare Davis, KIBA, BindingDB, and ChEMBL-derived data according to their respective licenses and the loaders/configurations in this package. Large pretrained molecular/protein assets and output predictions are also excluded.

## Claim Boundary

RCSDTA is a retrospective reliability and risk-aware triage framework. This release does not claim prospective wet-lab validation, universal temporal AURC gains, or a distribution-free guarantee under arbitrary data shifts.

## License

See [LICENSE](LICENSE).

