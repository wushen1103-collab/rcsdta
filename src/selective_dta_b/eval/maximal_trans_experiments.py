from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.ensemble import ExtraTreesRegressor

from selective_dta_b.eval.chembl_temporal_backtest import (
    _pairwise,
    _predict_ensemble,
    _prediction_frame,
    _risk_control_rows,
    _score_posthoc,
    _sequence_composition,
    _target_hash_features,
    _write_frame,
    add_target_familiarity,
    materialize_chembl_publication_year_split,
)
from selective_dta_b.eval.followup_experiments import discover_prediction_records, ensure_error_columns


RELEASE_BACKBONES = ("SimBoost", "DeepDTA", "GraphDTA", "KANPM", "ChemBERTaHybrid")
UTILITY_LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
UTILITY_BUDGETS = (10, 50)


def _parse_release_number(value: object) -> float:
    text = str(value)
    if "_" in text:
        text = text.rsplit("_", 1)[-1]
    try:
        return float(text)
    except ValueError:
        return float("nan")


def materialize_chembl_release_temporal_split(workspace: str | Path, *, output_dir: str | Path) -> tuple[pd.DataFrame, dict[str, object]]:
    root = Path(workspace).resolve()
    out = Path(output_dir).resolve()
    processed = root / "data" / "processed" / "chembl" / "standardized_pairs.csv"
    if not processed.exists():
        frame, _ = materialize_chembl_publication_year_split(root, output_dir=out, refresh=False)
    else:
        frame = pd.read_csv(processed)
    frame = frame.copy()
    frame["chembl_release_number"] = frame["chembl_release"].map(_parse_release_number)
    frame = frame.loc[frame["chembl_release_number"].notna()].copy()
    releases = sorted(frame["chembl_release_number"].dropna().unique())
    if len(releases) < 3:
        raise ValueError("ChEMBL release split needs at least three release groups")

    # The fetched ChEMBL cache naturally contains CHEMBL_1, CHEMBL_27/28, CHEMBL_32/33.
    # Use contiguous release blocks so the protocol is explicitly release-based.
    train_cut = releases[max(0, int(math.floor(len(releases) * 0.50)) - 1)]
    val_cut = releases[max(1, int(math.floor(len(releases) * 0.75)) - 1)]
    frame["split"] = "test"
    frame.loc[frame["chembl_release_number"] <= train_cut, "split"] = "train"
    frame.loc[(frame["chembl_release_number"] > train_cut) & (frame["chembl_release_number"] <= val_cut), "split"] = "val"
    frame["temporal_axis"] = "chembl_release"

    split_path = root / "data" / "processed" / "chembl" / "splits" / "release_temporal_seed42.csv"
    _write_frame(frame, split_path)
    _write_frame(frame, out / "chembl_release_temporal_pairs.csv")
    counts = frame["split"].value_counts().to_dict()
    status = {
        "temporal_axis": "chembl_release",
        "release_numbers": [int(x) for x in releases],
        "train_release_max": int(train_cut),
        "val_release_max": int(val_cut),
        "num_rows": int(len(frame)),
        "num_targets": int(frame["target_id"].nunique()),
        "num_molecules": int(frame["drug_id"].nunique()),
        "num_train": int(counts.get("train", 0)),
        "num_val": int(counts.get("val", 0)),
        "num_test": int(counts.get("test", 0)),
        "split_path": str(split_path),
    }
    return frame, status


def _chemberta_embeddings(smiles: pd.Series, *, workspace: Path, model_name: str = "seyonec/ChemBERTa-zinc-base-v1") -> pd.DataFrame:
    cache_dir = workspace / "data" / "external_temporal" / "chembl" / "foundation_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    names_path = cache_dir / "chemberta_smiles.csv"
    embeddings_path = cache_dir / "chemberta_embeddings.npy"
    requested = pd.Series(sorted(set(smiles.fillna("").astype(str))), name="drug_smiles")
    if names_path.exists() and embeddings_path.exists():
        names = pd.read_csv(names_path)
        emb = np.load(embeddings_path)
        if set(requested) <= set(names["drug_smiles"]) and len(names) == emb.shape[0]:
            out = names.copy()
            for idx in range(emb.shape[1]):
                out[f"chemberta_{idx}"] = emb[:, idx]
            return out.loc[out["drug_smiles"].isin(set(requested))].reset_index(drop=True)

    import torch
    from transformers import AutoModel, AutoTokenizer

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModel.from_pretrained(model_name, local_files_only=True).to(device)
    model.eval()
    vectors: list[np.ndarray] = []
    batch_size = 64 if device.type == "cuda" else 16
    with torch.no_grad():
        for start in range(0, len(requested), batch_size):
            batch = requested.iloc[start : start + batch_size].tolist()
            tokens = tokenizer(batch, padding=True, truncation=True, max_length=160, return_tensors="pt").to(device)
            output = model(**tokens).last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1).float()
            pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            vectors.append(pooled.detach().cpu().numpy().astype(np.float32))
    emb = np.vstack(vectors)
    requested.to_csv(names_path, index=False)
    np.save(embeddings_path, emb)
    out = requested.to_frame()
    for idx in range(emb.shape[1]):
        out[f"chemberta_{idx}"] = emb[:, idx]
    return out


def _chemberta_hybrid_features(frame: pd.DataFrame, *, workspace: Path) -> np.ndarray:
    emb = _chemberta_embeddings(frame["drug_smiles"], workspace=workspace)
    merged = frame[["drug_smiles", "target_id", "target_sequence", "target_familiarity", "target_novelty"]].merge(
        emb,
        on="drug_smiles",
        how="left",
    )
    emb_cols = [column for column in merged.columns if column.startswith("chemberta_")]
    chem = merged[emb_cols].fillna(0.0).to_numpy(dtype=np.float32)
    target_hash = _target_hash_features(merged["target_id"], n_features=96)
    seq_comp = _sequence_composition(merged["target_sequence"])
    fam = merged[["target_familiarity", "target_novelty"]].to_numpy(dtype=np.float32)
    return np.hstack([chem, target_hash, seq_comp, fam]).astype(np.float32)


def _predict_chemberta_hybrid(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    *,
    workspace: Path,
    ensemble_size: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train = _chemberta_hybrid_features(train, workspace=workspace)
    x_val = _chemberta_hybrid_features(val, workspace=workspace)
    x_test = _chemberta_hybrid_features(test, workspace=workspace)
    y = train["target"].to_numpy(dtype=float)
    rng = np.random.default_rng(random_state)
    val_preds: list[np.ndarray] = []
    test_preds: list[np.ndarray] = []
    for member in range(ensemble_size):
        idx = np.arange(len(train)) if member == 0 else rng.choice(len(train), size=len(train), replace=True)
        model = ExtraTreesRegressor(
            n_estimators=160,
            max_depth=26,
            min_samples_leaf=2,
            random_state=random_state + member,
            n_jobs=-1,
        )
        model.fit(x_train[idx], y[idx])
        val_preds.append(model.predict(x_val))
        test_preds.append(model.predict(x_test))
    val_stack = np.vstack(val_preds)
    test_stack = np.vstack(test_preds)
    return val_stack.mean(axis=0), val_stack.std(axis=0), test_stack.mean(axis=0), test_stack.std(axis=0)


def run_chembl_release_temporal_backtest(
    workspace: str | Path,
    *,
    output_dir: str | Path | None = None,
    ensemble_size: int = 3,
    random_state: int = 42,
) -> dict[str, object]:
    root = Path(workspace).resolve()
    out = Path(output_dir).resolve() if output_dir else root / "reports" / "trans_grade_experiments"
    out.mkdir(parents=True, exist_ok=True)
    frame, materialize_status = materialize_chembl_release_temporal_split(root, output_dir=out)
    frame = add_target_familiarity(frame)
    train = frame.loc[frame["split"] == "train"].copy()
    val = frame.loc[frame["split"] == "val"].copy()
    test = frame.loc[frame["split"] == "test"].copy()

    prediction_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    risk_rows: list[dict[str, object]] = []
    for backbone_name in RELEASE_BACKBONES:
        if backbone_name == "ChemBERTaHybrid":
            val_mean, val_std, test_mean, test_std = _predict_chemberta_hybrid(
                train,
                val,
                test,
                workspace=root,
                ensemble_size=ensemble_size,
                random_state=random_state,
            )
        else:
            val_mean, val_std, test_mean, test_std = _predict_ensemble(
                backbone_name,
                train,
                val,
                test,
                ensemble_size=ensemble_size,
                random_state=random_state,
            )
        val_pred = _prediction_frame(val, val_mean, val_std, backbone_name=backbone_name)
        test_pred = _prediction_frame(test, test_mean, test_std, backbone_name=backbone_name)
        scored = _score_posthoc(val_pred, test_pred, random_state=random_state)
        prediction_rows.append(scored)
        val_scored = _score_posthoc(val_pred, val_pred, random_state=random_state)
        meta = {
            "run_name": f"chembl_release_{backbone_name.lower()}_seed{random_state}",
            "dataset_name": "chembl",
            "split_name": "release_temporal",
            "seed": int(random_state),
            "model_type": backbone_name.lower(),
            "backbone_name": backbone_name,
            "backbone_protocol": "release_temporal_public_chembl_plus_posthoc_selector",
            "num_train": int(len(train)),
            "num_val": int(len(val)),
            "num_test": int(len(test)),
        }
        for source_name, confidence_col in (
            ("posthoc_selector", "confidence_posthoc"),
            ("mc_dropout", "confidence_mc_dropout"),
            ("target_familiarity", "target_familiarity"),
            ("oracle", "confidence_oracle"),
        ):
            if confidence_col in scored and scored[confidence_col].notna().sum() > 0:
                from selective_dta_b.eval.followup_experiments import selective_metrics

                summary_rows.append({**meta, "confidence_source": source_name, **selective_metrics(scored, confidence_col=confidence_col)})
                if source_name != "oracle":
                    source_val = val_scored if source_name == "posthoc_selector" else val_pred
                    risk_rows.extend(
                        _risk_control_rows(
                            source_val,
                            scored,
                            confidence_col=confidence_col,
                            source_name=source_name,
                            meta=meta,
                        )
                    )

    predictions = pd.concat(prediction_rows, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    pairwise = _pairwise(summary)
    risk = pd.DataFrame(risk_rows)
    _write_frame(predictions, out / "chembl_release_temporal_predictions.csv")
    _write_frame(summary, out / "chembl_release_temporal_summary.csv")
    _write_frame(pairwise, out / "chembl_release_temporal_pairwise_stats.csv")
    _write_frame(risk, out / "chembl_release_temporal_risk_control.csv")

    status = {
        **materialize_status,
        "status": "completed_release_temporal_backtest",
        "backbones": list(RELEASE_BACKBONES),
        "ensemble_size": int(ensemble_size),
        "summary_rows": int(len(summary)),
        "pairwise_rows": int(len(pairwise)),
        "risk_control_rows": int(len(risk)),
        "predictions_rows": int(len(predictions)),
    }
    (out / "chembl_release_temporal_status.json").write_text(json.dumps(status, indent=2))
    return status


def _pairwise_by_run(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric in ("aurc", "coverage_50_rmse", "coverage_70_rmse", "coverage_90_rmse"):
        if metric not in summary:
            continue
        pivot = summary.pivot_table(
            index=["run_name", "backbone_name", "seed"],
            columns="confidence_source",
            values=metric,
            aggfunc="first",
        ).dropna(how="all")
        for baseline in ("mc_dropout", "target_familiarity"):
            if "posthoc_selector" not in pivot or baseline not in pivot:
                continue
            paired = pivot[["posthoc_selector", baseline]].dropna()
            if paired.empty:
                continue
            delta = paired["posthoc_selector"] - paired[baseline]
            rows.append(
                {
                    "metric_name": metric,
                    "baseline_confidence_source": baseline,
                    "num_pairs": int(len(delta)),
                    "posthoc_mean": float(paired["posthoc_selector"].mean()),
                    "baseline_mean": float(paired[baseline].mean()),
                    "mean_delta_posthoc_minus_baseline": float(delta.mean()),
                    "posthoc_win_rate": float((delta < 0).mean()),
                }
            )
    return pd.DataFrame(rows)


def run_chembl_release_temporal_multiseed(
    workspace: str | Path,
    *,
    output_dir: str | Path | None = None,
    seeds: tuple[int, ...] = (42, 43, 44),
    ensemble_size: int = 3,
) -> dict[str, object]:
    root = Path(workspace).resolve()
    out = Path(output_dir).resolve() if output_dir else root / "reports" / "trans_grade_experiments"
    summaries: list[pd.DataFrame] = []
    risks: list[pd.DataFrame] = []
    predictions: list[pd.DataFrame] = []
    seed_status: dict[str, object] = {}
    for seed in seeds:
        seed_out = out / f"chembl_release_temporal_seed{seed}"
        status = run_chembl_release_temporal_backtest(
            root,
            output_dir=seed_out,
            ensemble_size=ensemble_size,
            random_state=seed,
        )
        seed_status[str(seed)] = status
        summary_path = seed_out / "chembl_release_temporal_summary.csv"
        risk_path = seed_out / "chembl_release_temporal_risk_control.csv"
        pred_path = seed_out / "chembl_release_temporal_predictions.csv"
        if summary_path.exists():
            frame = pd.read_csv(summary_path)
            frame["chembl_release_seed"] = int(seed)
            summaries.append(frame)
        if risk_path.exists():
            frame = pd.read_csv(risk_path)
            frame["chembl_release_seed"] = int(seed)
            risks.append(frame)
        if pred_path.exists():
            frame = pd.read_csv(pred_path)
            frame["chembl_release_seed"] = int(seed)
            predictions.append(frame)

    summary = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    risk = pd.concat(risks, ignore_index=True) if risks else pd.DataFrame()
    prediction = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    pairwise = _pairwise_by_run(summary) if not summary.empty else pd.DataFrame()
    _write_frame(summary, out / "chembl_release_temporal_multiseed_summary.csv")
    _write_frame(pairwise, out / "chembl_release_temporal_multiseed_pairwise_stats.csv")
    _write_frame(risk, out / "chembl_release_temporal_multiseed_risk_control.csv")
    _write_frame(prediction, out / "chembl_release_temporal_multiseed_predictions.csv")
    status = {
        "status": "completed_release_temporal_multiseed_backtest",
        "seeds": list(seeds),
        "ensemble_size": int(ensemble_size),
        "summary_rows": int(len(summary)),
        "pairwise_rows": int(len(pairwise)),
        "risk_control_rows": int(len(risk)),
        "predictions_rows": int(len(prediction)),
        "seed_status": seed_status,
    }
    (out / "chembl_release_temporal_multiseed_status.json").write_text(json.dumps(status, indent=2))
    return status


def _vs_metrics_for_group(group: pd.DataFrame, *, score_col: str, budget: int, active_fraction: float = 0.1) -> dict[str, object] | None:
    group = ensure_error_columns(group).loc[group["target"].notna() & group[score_col].notna()].copy()
    if len(group) < max(10, min(budget, 10)):
        return None
    n_active = max(1, int(math.ceil(len(group) * active_fraction)))
    active = set(group.sort_values("target", ascending=False).head(n_active)["row_id"])
    selected = group.sort_values(score_col, ascending=False).head(min(budget, len(group)))
    if selected.empty:
        return None
    hits = int(selected["row_id"].isin(active).sum())
    precision = hits / float(len(selected))
    active_rate = n_active / float(len(group))
    enrichment = precision / active_rate if active_rate else float("nan")
    mean_abs_error = float(selected["abs_error"].mean())
    return {
        "num_compounds": int(len(group)),
        "num_selected": int(len(selected)),
        "num_actives": int(n_active),
        "selected_hits": hits,
        "hit_recovery": float(hits / n_active),
        "precision_at_budget": precision,
        "false_positive_risk": float((len(selected) - hits) / len(selected)),
        "enrichment_factor": enrichment,
        "mean_selected_abs_error": mean_abs_error,
        "risk_adjusted_enrichment": float(enrichment / (1.0 + mean_abs_error)) if math.isfinite(enrichment) else float("nan"),
    }


def _utility_vs_rows(frame: pd.DataFrame, *, budget: int, lambda_value: float, method_name: str) -> list[dict[str, object]]:
    data = ensure_error_columns(frame).copy()
    if "target_id" not in data or "predicted_abs_error_posthoc" not in data:
        return []
    data["score_utility"] = data["prediction_mean"] - float(lambda_value) * data["predicted_abs_error_posthoc"]
    novelty = data.groupby("target_id")["target_novelty"].mean() if "target_novelty" in data else pd.Series(dtype=float)
    novelty_cutoff = float(novelty.quantile(0.75)) if len(novelty) else float("nan")
    rows: list[dict[str, object]] = []
    for target_id, group in data.groupby("target_id"):
        metrics = _vs_metrics_for_group(group, score_col="score_utility", budget=budget)
        if metrics is None:
            continue
        mean_novelty = float(novelty.get(target_id, float("nan"))) if len(novelty) else float("nan")
        rows.append(
            {
                "target_id": target_id,
                "decision_budget": int(budget),
                "decision_protocol": method_name,
                "utility_lambda": float(lambda_value),
                "mean_target_novelty": mean_novelty,
                "novel_target_subgroup": bool(math.isfinite(mean_novelty) and math.isfinite(novelty_cutoff) and mean_novelty >= novelty_cutoff),
                **metrics,
            }
        )
    return rows


def run_vs_utility_tuning(
    workspace: str | Path,
    *,
    output_dir: str | Path | None = None,
    paper_only: bool = True,
    max_runs: int | None = None,
) -> dict[str, object]:
    root = Path(workspace).resolve()
    out = Path(output_dir).resolve() if output_dir else root / "reports" / "trans_grade_experiments"
    records = discover_prediction_records(root, paper_only=paper_only, max_runs=max_runs)
    sensitivity_rows: list[dict[str, object]] = []
    target_rows: list[dict[str, object]] = []
    run_rows: list[dict[str, object]] = []

    for record in records:
        if record.validation_path is None or not record.validation_path.exists():
            continue
        val = pd.read_csv(record.validation_path)
        test = pd.read_csv(record.test_path)
        if "target_id" not in val or "target_id" not in test or "predicted_abs_error_posthoc" not in val or "predicted_abs_error_posthoc" not in test:
            continue
        meta = {
            "run_name": record.run_name,
            "dataset_name": record.dataset_name,
            "split_name": record.split_name,
            "seed": int(record.seed),
            "model_type": record.model_type,
        }
        for budget in UTILITY_BUDGETS:
            val_scores = []
            for lambda_value in UTILITY_LAMBDAS:
                rows = _utility_vs_rows(val, budget=budget, lambda_value=lambda_value, method_name="utility_lambda_grid")
                if not rows:
                    continue
                frame = pd.DataFrame(rows)
                score = float(frame["risk_adjusted_enrichment"].mean())
                val_scores.append((lambda_value, score))
                sensitivity_rows.append({**meta, "decision_budget": budget, "utility_lambda": lambda_value, "validation_mean_risk_adjusted_enrichment": score})
            if not val_scores:
                continue
            best_lambda, best_score = sorted(val_scores, key=lambda item: (-item[1], item[0]))[0]
            rows = _utility_vs_rows(test, budget=budget, lambda_value=best_lambda, method_name="posthoc_utility_tuned")
            rows = [{**meta, **row, "validation_selected_lambda": best_lambda, "validation_selected_score": best_score} for row in rows]
            target_rows.extend(rows)
            if rows:
                frame = pd.DataFrame(rows)
                run_rows.append(
                    {
                        **meta,
                        "decision_budget": int(budget),
                        "decision_protocol": "posthoc_utility_tuned",
                        "validation_selected_lambda": float(best_lambda),
                        "validation_selected_score": float(best_score),
                        "num_targets": int(frame["target_id"].nunique()),
                        "mean_hit_recovery": float(frame["hit_recovery"].mean()),
                        "mean_false_positive_risk": float(frame["false_positive_risk"].mean()),
                        "mean_risk_adjusted_enrichment": float(frame["risk_adjusted_enrichment"].mean()),
                        "mean_enrichment_factor": float(frame["enrichment_factor"].mean()),
                        "mean_precision_at_budget": float(frame["precision_at_budget"].mean()),
                        "mean_selected_abs_error": float(frame["mean_selected_abs_error"].mean()),
                        "novel_target_mean_risk_adjusted_enrichment": float(frame.loc[frame["novel_target_subgroup"], "risk_adjusted_enrichment"].mean()),
                    }
                )

    sensitivity = pd.DataFrame(sensitivity_rows)
    targets = pd.DataFrame(target_rows)
    runs = pd.DataFrame(run_rows)
    _write_frame(sensitivity, out / "vs_utility_lambda_sensitivity.csv")
    _write_frame(targets, out / "vs_utility_tuned_target_rows.csv")
    _write_frame(runs, out / "vs_utility_tuned_run_summary.csv")
    status = {
        "vs_utility_sensitivity_rows": int(len(sensitivity)),
        "vs_utility_target_rows": int(len(targets)),
        "vs_utility_run_rows": int(len(runs)),
    }
    (out / "vs_utility_tuning_status.json").write_text(json.dumps(status, indent=2))
    return status


def _holm_adjust(p_values: list[float]) -> list[float]:
    n = len(p_values)
    order = sorted(range(n), key=lambda idx: p_values[idx] if math.isfinite(p_values[idx]) else float("inf"))
    adjusted = [float("nan")] * n
    prev = 0.0
    for rank, idx in enumerate(order):
        p = p_values[idx]
        value = min(1.0, (n - rank) * p) if math.isfinite(p) else float("nan")
        if math.isfinite(value):
            value = max(value, prev)
            prev = value
        adjusted[idx] = value
    return adjusted


def _paired_rows(
    frame: pd.DataFrame,
    *,
    experiment_name: str,
    index_columns: list[str],
    method_column: str,
    treatment: str,
    baselines: tuple[str, ...],
    metrics: dict[str, str],
    bootstrap_reps: int = 2000,
) -> list[dict[str, object]]:
    rng = np.random.default_rng(20260521)
    rows: list[dict[str, object]] = []
    for metric_name, direction in metrics.items():
        if metric_name not in frame:
            continue
        pivot = frame.pivot_table(index=index_columns, columns=method_column, values=metric_name, aggfunc="first").dropna(how="all")
        for baseline in baselines:
            if treatment not in pivot or baseline not in pivot:
                continue
            paired = pivot[[treatment, baseline]].dropna()
            if len(paired) < 2:
                continue
            delta = paired[treatment].to_numpy(dtype=float) - paired[baseline].to_numpy(dtype=float)
            beneficial = delta < 0 if direction == "lower" else delta > 0
            boot = np.empty(bootstrap_reps, dtype=float)
            for idx in range(bootstrap_reps):
                sample = rng.choice(delta, size=len(delta), replace=True)
                boot[idx] = float(np.mean(sample))
            try:
                alternative = "less" if direction == "lower" else "greater"
                p_value = float(wilcoxon(delta, alternative=alternative, zero_method="zsplit").pvalue)
            except Exception:
                p_value = float("nan")
            rows.append(
                {
                    "experiment_name": experiment_name,
                    "metric_name": metric_name,
                    "direction": direction,
                    "treatment": treatment,
                    "baseline": baseline,
                    "num_pairs": int(len(delta)),
                    "treatment_mean": float(paired[treatment].mean()),
                    "baseline_mean": float(paired[baseline].mean()),
                    "mean_delta_treatment_minus_baseline": float(np.mean(delta)),
                    "delta_ci95_low": float(np.percentile(boot, 2.5)),
                    "delta_ci95_high": float(np.percentile(boot, 97.5)),
                    "treatment_win_rate": float(np.mean(beneficial)),
                    "wilcoxon_p_value": p_value,
                }
            )
    return rows


def run_paired_significance_package(workspace: str | Path, *, output_dir: str | Path | None = None) -> dict[str, object]:
    root = Path(workspace).resolve()
    out = Path(output_dir).resolve() if output_dir else root / "reports" / "trans_grade_experiments"
    rows: list[dict[str, object]] = []
    metric_lower = {"aurc": "lower", "coverage_50_rmse": "lower", "coverage_70_rmse": "lower", "coverage_90_rmse": "lower"}

    specs = [
        ("main_selective", root / "reports" / "summary" / "selective_paper_runs.csv", ["run_name"], "confidence_source", "posthoc_selector", ("mc_dropout", "target_familiarity"), metric_lower),
        ("modern_backbones", out / "modern_baseline_posthoc_summary.csv", ["run_name"], "confidence_source", "posthoc_selector", ("mc_dropout", "target_familiarity"), metric_lower),
        ("chembl_publication_year", out / "chembl_release_backtest_summary.csv", ["run_name"], "confidence_source", "posthoc_selector", ("mc_dropout", "target_familiarity"), metric_lower),
        ("chembl_release_temporal", out / "chembl_release_temporal_summary.csv", ["run_name"], "confidence_source", "posthoc_selector", ("mc_dropout", "target_familiarity"), metric_lower),
        ("chembl_release_temporal_multiseed", out / "chembl_release_temporal_multiseed_summary.csv", ["run_name", "chembl_release_seed"], "confidence_source", "posthoc_selector", ("mc_dropout", "target_familiarity"), metric_lower),
    ]
    for name, path, index_cols, method_col, treatment, baselines, metrics in specs:
        if path.exists():
            rows.extend(
                _paired_rows(
                    pd.read_csv(path),
                    experiment_name=name,
                    index_columns=index_cols,
                    method_column=method_col,
                    treatment=treatment,
                    baselines=baselines,
                    metrics=metrics,
                )
            )

    vs_path = out / "vs_decision_budget_run_summary.csv"
    utility_path = out / "vs_utility_tuned_run_summary.csv"
    if vs_path.exists() and utility_path.exists():
        vs = pd.read_csv(vs_path)
        util = pd.read_csv(utility_path)
        combined = pd.concat(
            [
                vs,
                util[[column for column in util.columns if column in vs.columns or column.startswith("mean_") or column in {"run_name", "decision_budget", "decision_protocol"}]],
            ],
            ignore_index=True,
            sort=False,
        )
        rows.extend(
            _paired_rows(
                combined,
                experiment_name="vs_decision_budget",
                index_columns=["run_name", "decision_budget"],
                method_column="decision_protocol",
                treatment="posthoc_utility_tuned",
                baselines=("prediction_only", "posthoc_lower_bound", "mc_weighted"),
                metrics={
                    "mean_risk_adjusted_enrichment": "higher",
                    "mean_hit_recovery": "higher",
                    "mean_false_positive_risk": "lower",
                    "mean_selected_abs_error": "lower",
                },
            )
        )

    risk_path = out / "formal_risk_control_run_rows.csv"
    if risk_path.exists():
        risk = pd.read_csv(risk_path)
        rows.extend(
            _paired_rows(
                risk,
                experiment_name="formal_risk_control",
                index_columns=["run_name", "risk_metric", "target_risk_threshold"],
                method_column="confidence_source",
                treatment="posthoc_selector",
                baselines=("mc_dropout", "target_familiarity"),
                metrics={"test_achieved_risk": "lower", "test_coverage": "higher"},
            )
        )

    stats = pd.DataFrame(rows)
    if not stats.empty:
        stats["holm_p_value"] = _holm_adjust(stats["wilcoxon_p_value"].tolist())
    _write_frame(stats, out / "paired_significance_summary.csv")
    status = {"paired_significance_rows": int(len(stats))}
    (out / "paired_significance_status.json").write_text(json.dumps(status, indent=2))
    return status


def run_maximal_trans_experiments(
    workspace: str | Path,
    *,
    output_dir: str | Path | None = None,
    sections: tuple[str, ...] = ("chembl_release", "vs_utility", "paired_stats"),
    max_runs: int | None = None,
) -> dict[str, object]:
    root = Path(workspace).resolve()
    out = Path(output_dir).resolve() if output_dir else root / "reports" / "trans_grade_experiments"
    out.mkdir(parents=True, exist_ok=True)
    status: dict[str, object] = {"workspace": str(root), "output_dir": str(out), "sections": list(sections)}
    if "chembl_release" in sections:
        status["chembl_release_temporal"] = run_chembl_release_temporal_backtest(root, output_dir=out)
    if "chembl_release_multiseed" in sections:
        status["chembl_release_temporal_multiseed"] = run_chembl_release_temporal_multiseed(root, output_dir=out)
    if "vs_utility" in sections:
        status["vs_utility_tuning"] = run_vs_utility_tuning(root, output_dir=out, max_runs=max_runs)
    if "paired_stats" in sections:
        status["paired_significance"] = run_paired_significance_package(root, output_dir=out)

    main_status_path = out / "status.json"
    if main_status_path.exists():
        main_status = json.loads(main_status_path.read_text())
    else:
        main_status = {}
    main_status["maximal_trans_experiments"] = status
    main_status_path.write_text(json.dumps(main_status, indent=2))
    (out / "maximal_trans_experiments_status.json").write_text(json.dumps(status, indent=2))
    return status


__all__ = [
    "run_chembl_release_temporal_backtest",
    "run_chembl_release_temporal_multiseed",
    "run_maximal_trans_experiments",
    "run_paired_significance_package",
    "run_vs_utility_tuning",
]
