from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


COVERAGE_LEVELS = (0.5, 0.7, 0.9, 0.95)
BASE4 = ("prediction_mean", "prediction_std_mc_dropout", "target_familiarity", "target_novelty")
FEATURE_SETS = {
    "mean_only": ("prediction_mean",),
    "mc_only": ("prediction_std_mc_dropout",),
    "target_familiarity_only": ("target_familiarity",),
    "target_novelty_only": ("target_novelty",),
    "mc_plus_target_novelty": ("prediction_std_mc_dropout", "target_familiarity", "target_novelty"),
    "base4": BASE4,
    "enriched9": BASE4 + ("mc_x_novelty", "mc_x_familiarity", "mean_x_novelty", "mc_sq", "novelty_sq"),
}


@dataclass(frozen=True)
class PredictionRecord:
    run_name: str
    dataset_name: str
    split_name: str
    seed: int
    model_type: str
    test_path: Path
    validation_path: Path | None


def ensure_error_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "target" not in out and "affinity_model_target" in out:
        out["target"] = out["affinity_model_target"]
    if "prediction" not in out and "prediction_mean" in out:
        out["prediction"] = out["prediction_mean"]
    if "abs_error" not in out:
        out["residual"] = out["prediction"] - out["target"]
        out["abs_error"] = out["residual"].abs()
    if "squared_error" not in out:
        out["squared_error"] = out["abs_error"] ** 2
    return out


def add_enriched_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["mc_x_novelty"] = out["prediction_std_mc_dropout"] * out["target_novelty"]
    out["mc_x_familiarity"] = out["prediction_std_mc_dropout"] * out["target_familiarity"]
    out["mean_x_novelty"] = out["prediction_mean"] * out["target_novelty"]
    out["mc_sq"] = out["prediction_std_mc_dropout"] ** 2
    out["novelty_sq"] = out["target_novelty"] ** 2
    return out


def _rmse(squared_error: pd.Series | np.ndarray) -> float:
    values = np.asarray(squared_error, dtype=float)
    return float(math.sqrt(np.mean(values))) if len(values) else float("nan")


def selective_metrics(frame: pd.DataFrame, *, confidence_col: str) -> dict[str, float | int]:
    data = ensure_error_columns(frame).sort_values(confidence_col, ascending=False).reset_index(drop=True)
    if len(data) == 0:
        return {"num_examples": 0}
    risks = data["squared_error"].expanding().mean().to_numpy(dtype=float)
    coverage = (np.arange(len(data), dtype=float) + 1.0) / float(len(data))
    out: dict[str, float | int] = {
        "num_examples": int(len(data)),
        "full_rmse": _rmse(data["squared_error"]),
        "full_mae": float(data["abs_error"].mean()),
        "aurc": float(np.trapz(risks, coverage)),
    }
    for level in COVERAGE_LEVELS:
        k = max(1, int(math.ceil(len(data) * level)))
        top = data.iloc[:k]
        key = int(round(level * 100))
        out[f"coverage_{key}_rmse"] = _rmse(top["squared_error"])
        out[f"coverage_{key}_mae"] = float(top["abs_error"].mean())
    return out


def summarize_abs_error_calibration(
    frame: pd.DataFrame,
    *,
    confidence_col: str,
    predicted_error_col: str | None = None,
    n_bins: int = 10,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    data = ensure_error_columns(frame)
    data = data.loc[data[confidence_col].notna()].copy()
    if len(data) == 0:
        return {"num_examples": 0, "ece_abs_error": float("nan")}, pd.DataFrame()
    q = min(n_bins, len(data))
    data["bin_index"] = pd.qcut(data[confidence_col].rank(method="first"), q=q, labels=False, duplicates="drop").astype(int)
    rows = []
    for idx, bucket in data.groupby("bin_index", sort=True):
        row = {
            "bin_index": int(idx),
            "num_examples": int(len(bucket)),
            "mean_confidence": float(bucket[confidence_col].mean()),
            "mean_abs_error": float(bucket["abs_error"].mean()),
            "rmse": _rmse(bucket["squared_error"]),
        }
        if predicted_error_col and predicted_error_col in bucket:
            row["mean_predicted_abs_error"] = float(bucket[predicted_error_col].mean())
            row["abs_error_calibration_gap"] = abs(row["mean_predicted_abs_error"] - row["mean_abs_error"])
        rows.append(row)
    bins = pd.DataFrame(rows)
    if "abs_error_calibration_gap" in bins:
        ece = float((bins["abs_error_calibration_gap"] * bins["num_examples"]).sum() / bins["num_examples"].sum())
    else:
        ece = float("nan")
    if data[confidence_col].nunique() > 1 and data["abs_error"].nunique() > 1:
        rho, p_value = spearmanr(data[confidence_col], -data["abs_error"])
    else:
        rho, p_value = float("nan"), float("nan")
    return {
        "num_examples": int(len(data)),
        "ece_abs_error": ece,
        "mean_abs_error": float(data["abs_error"].mean()),
        "rmse": _rmse(data["squared_error"]),
        "spearman_confidence_vs_negative_abs_error": float(rho),
        "spearman_p_value": float(p_value),
    }, bins


def build_temporal_proxy_split(frame: pd.DataFrame, *, train_fraction: float = 0.7, val_fraction: float = 0.1) -> pd.DataFrame:
    if train_fraction <= 0 or val_fraction <= 0 or train_fraction + val_fraction >= 1:
        raise ValueError("invalid split fractions")

    def key(value: object) -> tuple[int, str]:
        text = str(value)
        match = re.search(r"(\d+)$", text)
        return (int(match.group(1)) if match else 0, text)

    out = frame.copy().assign(_key=frame["row_id"].map(key)).sort_values("_key").drop(columns="_key").reset_index(drop=True)
    n = len(out)
    train_end = max(1, int(math.floor(n * train_fraction)))
    val_end = min(max(train_end + 1, int(math.floor(n * (train_fraction + val_fraction)))), n - 1)
    out["split"] = "test"
    out.loc[: train_end - 1, "split"] = "train"
    out.loc[train_end : val_end - 1, "split"] = "val"
    out["temporal_proxy_rank"] = np.arange(n, dtype=int)
    out["temporal_proxy_note"] = "row_id_order_proxy_no_publication_date"
    return out


def _ridge_model() -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])


def summarize_feature_ablation(
    validation: pd.DataFrame,
    test: pd.DataFrame,
    *,
    feature_sets: dict[str, tuple[str, ...]] | None = None,
    random_state: int = 42,
) -> list[dict[str, float | int | str]]:
    del random_state
    feature_sets = feature_sets or FEATURE_SETS
    val = add_enriched_features(ensure_error_columns(validation))
    tst = add_enriched_features(ensure_error_columns(test))
    rows = []
    for name, cols in feature_sets.items():
        if any(col not in val or col not in tst for col in cols):
            continue
        train = val.dropna(subset=list(cols) + ["abs_error"])
        score = tst.dropna(subset=list(cols)).copy()
        if len(train) < 2 or len(score) == 0:
            continue
        model = _ridge_model()
        model.fit(train.loc[:, cols], train["abs_error"])
        pred_error = np.clip(model.predict(score.loc[:, cols]), a_min=0.0, a_max=None)
        score["confidence_ablation"] = 1.0 / (1.0 + pred_error)
        rows.append({"feature_variant": name, "regressor_type": "ridge", **selective_metrics(score, confidence_col="confidence_ablation")})
    return rows


def summarize_virtual_screening_group(
    frame: pd.DataFrame,
    *,
    score_col: str,
    active_fraction: float = 0.1,
    top_fraction: float = 0.05,
    method_name: str,
) -> dict[str, float | int | str]:
    data = frame.loc[frame[score_col].notna() & frame["target"].notna()].copy()
    if len(data) == 0:
        return {"method": method_name, "num_compounds": 0, "num_selected": 0, "num_actives": 0}
    n_active = max(1, int(math.ceil(len(data) * active_fraction)))
    n_select = max(1, int(math.ceil(len(data) * top_fraction)))
    active = set(data.sort_values("target", ascending=False).head(n_active)["row_id"])
    selected = data.sort_values(score_col, ascending=False).head(n_select)
    hits = int(selected["row_id"].isin(active).sum())
    precision = hits / float(n_select)
    active_rate = n_active / float(len(data))
    return {
        "method": method_name,
        "num_compounds": int(len(data)),
        "num_selected": int(n_select),
        "num_actives": int(n_active),
        "selected_hits": hits,
        "precision_at_k": float(precision),
        "active_rate": float(active_rate),
        "enrichment_factor": float(precision / active_rate) if active_rate else float("nan"),
    }


def _summary_table(workspace: Path, paper_only: bool) -> pd.DataFrame:
    name = "selective_paper_runs.csv" if paper_only else "selective_all_runs.csv"
    path = workspace / "reports" / "summary" / name
    if not path.exists():
        path = workspace / "reports" / "summary" / "selective_all_runs.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _infer_seed(run_name: str) -> int:
    match = re.search(r"seed(\d+)", run_name)
    return int(match.group(1)) if match else 42


def _infer_model(run_name: str) -> str:
    if run_name.startswith("char_baseline"):
        return "baseline"
    if run_name.startswith("char_heteroscedastic"):
        return "heteroscedastic"
    for name in ("deepdtagen", "deepdta", "graphdta", "moltrans", "kanpm", "simboost", "adambind"):
        if run_name.startswith(name):
            return name
    return "unknown"


def _infer_split(run_name: str) -> str:
    for name in ("similarity_aware_unseen_target", "unseen_target", "unseen_drug", "all_unseen", "random", "temporal_proxy"):
        if name in run_name:
            return name
    return "unknown"


def discover_prediction_records(workspace: str | Path, *, paper_only: bool = True, max_runs: int | None = None) -> list[PredictionRecord]:
    root = Path(workspace).resolve()
    table = _summary_table(root, paper_only)
    meta = {}
    allowed = None
    if not table.empty:
        sub = table[(table["evaluation_kind"] == "posthoc_selector") & (table["confidence_source"] == "posthoc_selector")]
        sub = sub.drop_duplicates("run_name")
        allowed = set(sub["run_name"]) if paper_only else None
        meta = {str(r["run_name"]): r for r in sub.to_dict("records")}
    records = []
    for test_path in sorted(root.rglob("posthoc_selector/*_test_predictions.csv")):
        run_name = test_path.name[: -len("_test_predictions.csv")]
        if allowed is not None and run_name not in allowed:
            continue
        row = meta.get(run_name)
        if row:
            dataset, split, seed, model = str(row["dataset_name"]), str(row["split_name"]), int(row["seed"]), str(row["model_type"])
        else:
            tiny = pd.read_csv(test_path, nrows=1)
            dataset, split, seed, model = str(tiny["dataset_name"].iloc[0]), _infer_split(run_name), _infer_seed(run_name), _infer_model(run_name)
        val_path = test_path.with_name(f"{run_name}_validation_predictions.csv")
        records.append(PredictionRecord(run_name, dataset, split, seed, model, test_path, val_path if val_path.exists() else None))
    return records[:max_runs] if max_runs else records


def _read_frame(path: Path | None, compact: bool = True) -> pd.DataFrame | None:
    if path is None:
        return None
    if not compact:
        return ensure_error_columns(pd.read_csv(path))
    wanted = [
        "row_id", "dataset_name", "drug_id", "drug_smiles", "target_id", "target_sequence",
        "target", "prediction_mean", "prediction_std_mc_dropout", "target_familiarity",
        "target_novelty", "predicted_abs_error_posthoc", "confidence_posthoc",
        "confidence_mc_dropout", "abs_error", "squared_error",
    ]
    cols = pd.read_csv(path, nrows=0).columns
    return ensure_error_columns(pd.read_csv(path, usecols=[c for c in wanted if c in cols]))


def _payload(record: PredictionRecord) -> dict[str, object]:
    return {
        "run_name": record.run_name,
        "dataset_name": record.dataset_name,
        "split_name": record.split_name,
        "seed": record.seed,
        "model_type": record.model_type,
    }


def _sources(frame: pd.DataFrame) -> list[tuple[str, str, str | None]]:
    out = []
    if "confidence_posthoc" in frame:
        out.append(("posthoc_selector", "confidence_posthoc", "predicted_abs_error_posthoc"))
    if "confidence_mc_dropout" in frame:
        out.append(("mc_dropout", "confidence_mc_dropout", None))
    if "target_familiarity" in frame:
        out.append(("target_familiarity", "target_familiarity", None))
    return out


def _write(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _risk_rows(validation: pd.DataFrame | None, test: pd.DataFrame, confidence_col: str) -> list[dict[str, object]]:
    test_metrics = selective_metrics(test, confidence_col=confidence_col)
    val_metrics = selective_metrics(validation, confidence_col=confidence_col) if validation is not None and confidence_col in validation else {}
    rows = []
    for level in COVERAGE_LEVELS:
        key = int(round(level * 100))
        test_rmse = float(test_metrics.get(f"coverage_{key}_rmse", float("nan")))
        val_rmse = float(val_metrics.get(f"coverage_{key}_rmse", float("nan")))
        gap = test_rmse - val_rmse if math.isfinite(val_rmse) else float("nan")
        rows.append({"coverage": level, "test_rmse": test_rmse, "validation_rmse": val_rmse, "test_minus_validation_rmse": gap, "violates_validation_bound": bool(math.isfinite(gap) and gap > 0)})
    return rows


def _rdkit_ready() -> bool:
    try:
        import rdkit  # noqa: F401
    except Exception:
        return False
    return True


def compute_drug_novelty_cache(split_frame: pd.DataFrame, cache_path: Path) -> pd.DataFrame:
    if cache_path.exists():
        return pd.read_csv(cache_path)
    rows = split_frame.loc[split_frame["split"] == "test", ["row_id", "drug_id", "drug_smiles"]].copy()
    rows["drug_max_train_tanimoto"] = np.nan
    rows["drug_novelty"] = np.nan
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if not _rdkit_ready():
        rows.to_csv(cache_path, index=False)
        return rows
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    def fp(smiles: object):
        mol = Chem.MolFromSmiles(str(smiles))
        return None if mol is None else gen.GetFingerprint(mol)

    train_fps = [x for x in split_frame.loc[split_frame["split"] == "train", "drug_smiles"].drop_duplicates().map(fp) if x is not None]
    if train_fps:
        lookup = {}
        for item in rows[["drug_id", "drug_smiles"]].drop_duplicates("drug_id").itertuples(index=False):
            item_fp = fp(item.drug_smiles)
            lookup[item.drug_id] = 0.0 if item_fp is None else float(max(DataStructs.BulkTanimotoSimilarity(item_fp, train_fps)))
        rows["drug_max_train_tanimoto"] = rows["drug_id"].map(lookup)
        rows["drug_novelty"] = 1.0 - rows["drug_max_train_tanimoto"]
    rows.to_csv(cache_path, index=False)
    return rows


def compute_activity_cliff_cache(split_frame: pd.DataFrame, cache_path: Path, *, tanimoto_threshold: float = 0.7, affinity_delta_threshold: float = 1.0) -> pd.DataFrame:
    if cache_path.exists():
        return pd.read_csv(cache_path)
    test = split_frame.loc[split_frame["split"] == "test", ["row_id", "target_id", "drug_smiles", "affinity_model_target"]].copy()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if not _rdkit_ready():
        test[["activity_cliff_max_train_tanimoto", "activity_cliff_max_affinity_delta"]] = np.nan
        test["activity_cliff_flag"] = False
        test.to_csv(cache_path, index=False)
        return test
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    def fp(smiles: object):
        mol = Chem.MolFromSmiles(str(smiles))
        return None if mol is None else gen.GetFingerprint(mol)

    train = split_frame.loc[split_frame["split"] == "train", ["target_id", "drug_smiles", "affinity_model_target"]].copy()
    train["_fp"] = train["drug_smiles"].map(fp)
    groups = {k: g.dropna(subset=["_fp"]) for k, g in train.groupby("target_id")}
    rows = []
    for row in test.itertuples(index=False):
        item_fp = fp(row.drug_smiles)
        group = groups.get(row.target_id)
        if item_fp is None or group is None or len(group) == 0:
            max_sim, max_delta, flag = float("nan"), float("nan"), False
        else:
            sims = DataStructs.BulkTanimotoSimilarity(item_fp, list(group["_fp"]))
            idx = [i for i, s in enumerate(sims) if s >= tanimoto_threshold]
            max_sim = float(max(sims)) if sims else float("nan")
            if idx:
                deltas = (group.iloc[idx]["affinity_model_target"].astype(float) - float(row.affinity_model_target)).abs()
                max_delta = float(deltas.max())
                flag = bool(max_delta >= affinity_delta_threshold)
            else:
                max_delta, flag = float("nan"), False
        rows.append({"row_id": row.row_id, "activity_cliff_max_train_tanimoto": max_sim, "activity_cliff_max_affinity_delta": max_delta, "activity_cliff_flag": flag})
    out = pd.DataFrame(rows)
    out.to_csv(cache_path, index=False)
    return out


def _bin(series: pd.Series, n_bins: int = 5) -> pd.Series:
    valid = series.dropna()
    if valid.empty:
        return pd.Series(["missing"] * len(series), index=series.index)
    if valid.nunique(dropna=True) <= 1:
        result = pd.Series(["missing"] * len(series), index=series.index)
        result.loc[valid.index] = "single_bin"
        return result

    labels = pd.qcut(
        valid.rank(method="first"),
        q=min(n_bins, valid.nunique(dropna=True)),
        labels=False,
        duplicates="drop",
    ).map(lambda x: f"q{int(x)+1}")
    result = pd.Series(["missing"] * len(series), index=series.index)
    result.loc[labels.index] = labels
    return result


def run_followup_experiments(workspace: str | Path, *, output_dir: str | Path | None = None, paper_only: bool = True, max_runs: int | None = None, min_compounds_per_target: int = 10) -> dict[str, object]:
    root = Path(workspace).resolve()
    out = Path(output_dir).resolve() if output_dir else root / "reports" / "followup_experiments"
    out.mkdir(parents=True, exist_ok=True)
    records = discover_prediction_records(root, paper_only=paper_only, max_runs=max_runs)
    (out / "manifest.json").write_text(json.dumps({"num_records": len(records), "paper_only": paper_only, "records": [{**_payload(r), "test_path": str(r.test_path), "validation_path": str(r.validation_path) if r.validation_path else None} for r in records]}, indent=2))

    cal_rows: list[dict[str, object]] = []
    bin_rows: list[dict[str, object]] = []
    risk_rows: list[dict[str, object]] = []
    novelty_rows: list[dict[str, object]] = []
    ablation_rows: list[dict[str, object]] = []
    cliff_rows: list[dict[str, object]] = []
    vs_group_rows: list[dict[str, object]] = []
    vs_run_rows: list[dict[str, object]] = []
    val_frames: list[pd.DataFrame] = []
    test_frames: dict[str, pd.DataFrame] = {}

    for record in records:
        test = _read_frame(record.test_path)
        validation = _read_frame(record.validation_path)
        if test is None:
            continue
        payload = _payload(record)

        for label, conf_col, pred_col in _sources(test):
            summary, bins = summarize_abs_error_calibration(test, confidence_col=conf_col, predicted_error_col=pred_col)
            cal_rows.append({**payload, "confidence_source": label, **summary})
            bin_rows.extend({**payload, "confidence_source": label, **row} for row in bins.to_dict("records"))
            risk_rows.extend({**payload, "confidence_source": label, **row} for row in _risk_rows(validation, test, conf_col))

        split_path = root / "data" / "processed" / record.dataset_name / "splits" / f"{record.split_name}_seed{record.seed}.csv"
        if split_path.exists():
            split = pd.read_csv(split_path)
            drug_novelty = compute_drug_novelty_cache(split, out / "cache" / f"drug_novelty_{record.dataset_name}_{record.split_name}_seed{record.seed}.csv")
            test = test.merge(drug_novelty[["row_id", "drug_max_train_tanimoto", "drug_novelty"]], on="row_id", how="left")
            cliff = compute_activity_cliff_cache(split, out / "cache" / f"activity_cliffs_{record.dataset_name}_{record.split_name}_seed{record.seed}.csv")
            test = test.merge(cliff, on="row_id", how="left")

        for novelty_col in [c for c in ("target_novelty", "drug_novelty") if c in test]:
            test[f"{novelty_col}_bin"] = _bin(test[novelty_col])
            for label, conf_col, _ in _sources(test):
                for name, bucket in test.groupby(f"{novelty_col}_bin", dropna=False):
                    novelty_rows.append({**payload, "confidence_source": label, "novelty_type": novelty_col, "novelty_bin": str(name), "mean_novelty": float(bucket[novelty_col].mean()) if bucket[novelty_col].notna().any() else float("nan"), **selective_metrics(bucket, confidence_col=conf_col)})

        if validation is not None:
            ablation_rows.extend({**payload, **row} for row in summarize_feature_ablation(validation, test))
            if "confidence_posthoc" in test:
                ablation_rows.append({**payload, "feature_variant": "existing_posthoc_selector", "regressor_type": "stored_model", **selective_metrics(test, confidence_col="confidence_posthoc")})
            val_frames.append(_compact(record, validation))
            test_frames[record.run_name] = _compact(record, test)

        if "activity_cliff_flag" in test:
            test["activity_cliff_flag"] = test["activity_cliff_flag"].fillna(False).astype(bool)
            for label, conf_col, _ in _sources(test):
                for flag, bucket in test.groupby("activity_cliff_flag"):
                    cliff_rows.append({**payload, "confidence_source": label, "activity_cliff_flag": bool(flag), "mean_max_train_tanimoto": float(bucket["activity_cliff_max_train_tanimoto"].mean()), "mean_max_affinity_delta": float(bucket["activity_cliff_max_affinity_delta"].mean()), **selective_metrics(bucket, confidence_col=conf_col)})

        run_vs = []
        if "target_id" in test:
            scores = {"prediction_only": "prediction_mean"}
            if "confidence_posthoc" in test:
                test["score_posthoc_weighted"] = test["prediction_mean"] * test["confidence_posthoc"]
                scores["posthoc_weighted"] = "score_posthoc_weighted"
                if "predicted_abs_error_posthoc" in test:
                    test["score_posthoc_lower_bound"] = test["prediction_mean"] - test["predicted_abs_error_posthoc"]
                    scores["posthoc_lower_bound"] = "score_posthoc_lower_bound"
            if "confidence_mc_dropout" in test:
                test["score_mc_weighted"] = test["prediction_mean"] * test["confidence_mc_dropout"]
                scores["mc_weighted"] = "score_mc_weighted"
            for target_id, group in test.groupby("target_id"):
                if len(group) < min_compounds_per_target:
                    continue
                for method, score_col in scores.items():
                    row = {**payload, "target_id": target_id, **summarize_virtual_screening_group(group, score_col=score_col, method_name=method)}
                    vs_group_rows.append(row)
                    run_vs.append(row)
        if run_vs:
            frame = pd.DataFrame(run_vs)
            for method, bucket in frame.groupby("method"):
                vs_run_rows.append({**payload, "method": method, "num_targets": int(len(bucket)), "mean_enrichment_factor": float(bucket["enrichment_factor"].mean()), "median_enrichment_factor": float(bucket["enrichment_factor"].median()), "mean_precision_at_k": float(bucket["precision_at_k"].mean())})

    transfer_rows = _transfer_rows(val_frames, test_frames)
    temporal_status = _temporal_outputs(root, out)

    _write(cal_rows, out / "calibration_run_summary.csv")
    _write(bin_rows, out / "calibration_bins.csv")
    _write(risk_rows, out / "risk_control_coverage.csv")
    _write(novelty_rows, out / "novelty_bin_summary.csv")
    _write(ablation_rows, out / "feature_ablation_summary.csv")
    _write(transfer_rows, out / "transfer_selector_summary.csv")
    _write(cliff_rows, out / "activity_cliff_summary.csv")
    _write(vs_group_rows, out / "virtual_screening_target_summary.csv")
    _write(vs_run_rows, out / "virtual_screening_run_summary.csv")

    status = {
        "num_records": len(records),
        "calibration_rows": len(cal_rows),
        "novelty_rows": len(novelty_rows),
        "feature_ablation_rows": len(ablation_rows),
        "transfer_rows": len(transfer_rows),
        "activity_cliff_rows": len(cliff_rows),
        "virtual_screening_target_rows": len(vs_group_rows),
        "virtual_screening_run_rows": len(vs_run_rows),
        **temporal_status,
    }
    (out / "status.json").write_text(json.dumps(status, indent=2))
    return status


def _compact(record: PredictionRecord, frame: pd.DataFrame) -> pd.DataFrame:
    data = add_enriched_features(ensure_error_columns(frame))
    keep = ["row_id", "target", "prediction_mean", "prediction_std_mc_dropout", "target_familiarity", "target_novelty", "abs_error", "squared_error", *FEATURE_SETS["enriched9"]]
    data = data[[c for c in dict.fromkeys(keep) if c in data]].copy()
    for key, value in _payload(record).items():
        data[key] = value
    return data


def _score_transfer(source: pd.DataFrame, target: pd.DataFrame, label: str) -> dict[str, object] | None:
    cols = FEATURE_SETS["enriched9"]
    src = source.dropna(subset=list(cols) + ["abs_error"])
    tgt = target.dropna(subset=list(cols)).copy()
    if len(src) < 10 or len(tgt) == 0:
        return None
    model = _ridge_model()
    model.fit(src.loc[:, cols], src["abs_error"])
    pred_error = np.clip(model.predict(tgt.loc[:, cols]), a_min=0.0, a_max=None)
    tgt["confidence_transfer"] = 1.0 / (1.0 + pred_error)
    return {"transfer_setting": label, "num_transfer_train_examples": int(len(src)), **selective_metrics(tgt, confidence_col="confidence_transfer")}


def _transfer_rows(val_frames: list[pd.DataFrame], test_frames: dict[str, pd.DataFrame]) -> list[dict[str, object]]:
    if not val_frames:
        return []
    validation = pd.concat(val_frames, ignore_index=True)
    rows = []
    for run_name, target in test_frames.items():
        meta = {k: target[k].iloc[0] for k in ["run_name", "dataset_name", "split_name", "seed", "model_type"]}
        source = validation[(validation["model_type"] == meta["model_type"]) & (validation["split_name"] == meta["split_name"]) & (validation["dataset_name"] != meta["dataset_name"])]
        result = _score_transfer(source, target, "same_model_other_datasets")
        if result:
            rows.append({**meta, **result})
        source = validation[(validation["dataset_name"] == meta["dataset_name"]) & (validation["split_name"] == meta["split_name"]) & (validation["model_type"] != meta["model_type"])]
        result = _score_transfer(source, target, "other_models_same_dataset")
        if result:
            rows.append({**meta, **result})
    return rows


def _temporal_outputs(root: Path, out: Path) -> dict[str, object]:
    audit_rows = []
    split_rows = []
    patterns = ("date", "year", "pub", "doi", "pmid", "release", "created", "updated")
    for dataset in ("bindingdb", "davis", "kiba"):
        path = root / "data" / "processed" / dataset / "standardized_pairs.csv"
        if not path.exists():
            continue
        columns = pd.read_csv(path, nrows=0).columns.tolist()
        date_like = [c for c in columns if any(p in c.lower() for p in patterns)]
        audit_rows.append({"dataset_name": dataset, "file_kind": "processed_standardized", "path": str(path), "num_columns": len(columns), "date_like_columns": ",".join(date_like), "has_temporal_metadata": bool(date_like)})
        frame = pd.read_csv(path)
        split = build_temporal_proxy_split(frame)
        split_path = root / "data" / "processed" / dataset / "splits" / "temporal_proxy_seed42.csv"
        split_path.parent.mkdir(parents=True, exist_ok=True)
        split.to_csv(split_path, index=False)
        counts = split["split"].value_counts().to_dict()
        split_rows.append({"dataset_name": dataset, "split_name": "temporal_proxy", "seed": 42, "split_path": str(split_path), "num_rows": len(split), "num_train": counts.get("train", 0), "num_val": counts.get("val", 0), "num_test": counts.get("test", 0), "note": "proxy uses row_id order because no reliable publication date column exists"})
        for raw in (root / "data").glob(f"**/{dataset}*"):
            if raw.is_file() and raw.suffix.lower() in {".csv", ".tsv", ".tab"}:
                try:
                    raw_cols = pd.read_csv(raw, sep=None, engine="python", nrows=0).columns.tolist()
                except Exception:
                    continue
                raw_date = [c for c in raw_cols if any(p in c.lower() for p in patterns)]
                audit_rows.append({"dataset_name": dataset, "file_kind": "raw_or_intermediate", "path": str(raw), "num_columns": len(raw_cols), "date_like_columns": ",".join(raw_date), "has_temporal_metadata": bool(raw_date)})
    _write(audit_rows, out / "temporal_metadata_audit.csv")
    _write(split_rows, out / "temporal_proxy_split_audit.csv")
    (out / "temporal_prospective_blueprint.json").write_text(json.dumps({"status": "proxy_splits_created", "limitation": "workspace data lacks reliable date/publication-year metadata", "recommended_true_temporal_sources": ["BindingDB entry metadata", "ChEMBL document/year metadata"], "proxy_split_name": "temporal_proxy", "proxy_seed": 42}, indent=2))
    return {"temporal_audit_rows": len(audit_rows), "temporal_proxy_splits": len(split_rows)}


__all__ = [
    "PredictionRecord",
    "build_temporal_proxy_split",
    "compute_activity_cliff_cache",
    "compute_drug_novelty_cache",
    "discover_prediction_records",
    "run_followup_experiments",
    "selective_metrics",
    "summarize_abs_error_calibration",
    "summarize_feature_ablation",
    "summarize_virtual_screening_group",
]
