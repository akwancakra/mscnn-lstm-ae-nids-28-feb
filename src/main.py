"""Main orchestration: full pipeline from data loading to evaluation.

Usage:
    python -m src.main --config src/config.yaml

Stages:
    1. Load & preprocess benign CIC-IDS-2017 data
    2. Domain shift analysis (CIC vs CSE)
    3. Train Stage 1 MSCNN-AE
    4. Extract latent vectors, analyze sessions, create windows
    5. Train Stage 2 BiLSTM-AE
    6. Compute combined anomaly scores & thresholds
    7. Evaluate on CIC-2017 all-label + CSE-2018 all-label
    8. Generate report
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import (
    ensure_dir,
    get_path,
    load_npz,
    load_yaml,
    resolve_paths,
    save_json,
    save_npz,
    set_global_seed,
    setup_gpu,
    setup_logging,
)

logger = logging.getLogger(__name__)


def run_pipeline(cfg: dict) -> dict:
    """Execute the complete two-stage pipeline."""

    from src.data.loader import (
        build_column_mapper,
        compute_shared_features,
        get_reference_columns,
        list_csv_files,
    )
    from src.data.preprocessing import (
        PreprocessingPipeline,
        load_all_labeled,
        load_and_prepare_benign_train,
    )
    from src.data.domain_shift import (
        ks_test_per_feature,
        plot_domain_shift,
        summarize_shift,
    )
    from src.data.windowing import (
        analyze_session_lengths,
        create_windows,
        plot_session_lengths,
    )
    from src.model.lstm_ae import compute_temporal_latent_dim
    from src.training.trainer import (
        compute_stage1_errors,
        compute_stage2_errors,
        extract_latent_vectors,
        train_stage1,
        train_stage2,
    )
    from src.training.threshold import (
        combine_scores,
        compute_all_thresholds,
        normalize_errors,
    )
    from src.evaluation.metrics import (
        binary_labels,
        build_threshold_comparison_table,
        compute_metrics,
        compute_pr_curve,
        compute_roc_curve,
        per_attack_detection_rate,
    )
    from src.evaluation.visualization import (
        plot_confusion_matrix,
        plot_error_distribution,
        plot_error_violin,
        plot_per_attack_dr,
        plot_pr_curves,
        plot_roc_curves,
        plot_threshold_comparison,
    )

    t0 = time.time()
    report = {}

    # ============================================================
    # 0. Setup
    # ============================================================
    cfg = resolve_paths(cfg)
    seed = cfg.get("runtime", {}).get("random_seed", 42)
    set_global_seed(seed)
    setup_gpu(cfg)

    models_dir = str(ensure_dir(get_path(cfg, "models_dir")))
    results_dir = str(ensure_dir(get_path(cfg, "results_dir")))
    processed_dir = str(ensure_dir(get_path(cfg, "data_processed")))

    pp_cfg = cfg.get("preprocessing", {})
    split_cfg = cfg.get("split", {})
    window_cfg = cfg.get("windowing", {})
    scoring_cfg = cfg.get("scoring", {})

    # ============================================================
    # 1. Discover datasets
    # ============================================================
    logger.info("=" * 60)
    logger.info("STAGE 1: DATA DISCOVERY & PREPROCESSING")
    logger.info("=" * 60)

    cic_files = list_csv_files(get_path(cfg, "data_raw_cic"))
    cse_files = list_csv_files(get_path(cfg, "data_raw_cse"))

    if not cic_files:
        raise FileNotFoundError(f"No CIC-IDS-2017 CSV files in {get_path(cfg, 'data_raw_cic')}")
    if not cse_files:
        raise FileNotFoundError(f"No CSE-CIC-IDS-2018 CSV files in {get_path(cfg, 'data_raw_cse')}")

    drop_columns = pp_cfg.get("drop_columns", ["Flow ID"])
    label_candidates = pp_cfg.get("label_candidates", ["Label"])
    benign_label = pp_cfg.get("benign_label", "BENIGN")
    session_cfg = pp_cfg.get("session_columns", {})

    shared_features, cic_label, cse_label, cse_mapper = compute_shared_features(
        cic_files, cse_files, drop_columns, label_candidates,
    )
    report["n_shared_features"] = len(shared_features)
    logger.info("Shared features: %d", len(shared_features))

    skip_preprocessing = cfg.get("runtime", {}).get("skip_preprocessing", False)
    processed_path = Path(processed_dir)
    pipeline_path = processed_path / "pipeline.joblib"
    cache_npz_path = processed_path / "train_val_2d.npz"
    meta_train_path = processed_path / "meta_train.parquet"
    meta_val_path = processed_path / "meta_val.parquet"

    if skip_preprocessing:
        # Load pipeline and cached arrays/meta; skip raw load and fit
        if not pipeline_path.exists():
            raise FileNotFoundError(
                "skip_preprocessing requested but pipeline not found. "
                f"Run once with skip_preprocessing=false to create {pipeline_path}"
            )
        if not cache_npz_path.exists():
            raise FileNotFoundError(
                "skip_preprocessing requested but train_val_2d.npz not found. "
                f"Run once with skip_preprocessing=false to create {cache_npz_path}"
            )
        if not meta_train_path.exists() or not meta_val_path.exists():
            raise FileNotFoundError(
                "skip_preprocessing requested but meta_train.parquet or meta_val.parquet not found. "
                "Run once with skip_preprocessing=false to create cache."
            )
        pipeline = PreprocessingPipeline(cfg).load(pipeline_path)
        cache = load_npz(cache_npz_path)
        X_train = cache["X_train"]
        X_val = cache["X_val"]
        meta_train = pd.read_parquet(meta_train_path)
        meta_val = pd.read_parquet(meta_val_path)
        report["n_benign_train"] = len(X_train)
        report["n_benign_val"] = len(X_val)
        report["n_features_original"] = pipeline.n_features_original
        report["n_features_final"] = pipeline.n_features_final
        report["reshape_2d"] = (pipeline.nx, pipeline.ny)
        report["latent_dim"] = pipeline.latent_dim
        report["feature_names"] = pipeline.feature_names
        logger.info("Loaded cached preprocessing: pipeline, X_train %s, X_val %s", X_train.shape, X_val.shape)
    else:
        # ============================================================
        # 2. Load benign CIC-2017 train/val
        # ============================================================
        X_train_raw, X_val_raw, meta_train, meta_val = load_and_prepare_benign_train(
            cic_files, shared_features, cic_label, benign_label, session_cfg,
            chunksize=pp_cfg.get("chunksize", 50000),
            val_size=split_cfg.get("val_size", 0.2),
            split_by_file=split_cfg.get("split_by_file", True),
        )
        report["n_benign_train"] = len(X_train_raw)
        report["n_benign_val"] = len(X_val_raw)

        # ============================================================
        # 3. Fit preprocessing pipeline
        # ============================================================
        pipeline = PreprocessingPipeline(cfg)
        pipeline.fit(X_train_raw, shared_features)
        pipeline.save(pipeline_path)

        report["n_features_original"] = pipeline.n_features_original
        report["n_features_final"] = pipeline.n_features_final
        report["reshape_2d"] = (pipeline.nx, pipeline.ny)
        report["latent_dim"] = pipeline.latent_dim
        report["feature_names"] = pipeline.feature_names

        X_train = pipeline.transform(X_train_raw, reshape_2d=True)
        X_val = pipeline.transform(X_val_raw, reshape_2d=True)
        logger.info("Train shape: %s, Val shape: %s", X_train.shape, X_val.shape)

        save_npz(cache_npz_path, X_train=X_train, X_val=X_val)
        meta_train.to_parquet(meta_train_path, index=False)
        meta_val.to_parquet(meta_val_path, index=False)
        logger.info("Saved preprocessing cache: train_val_2d.npz, meta_train/val.parquet")

    # ============================================================
    # 4. Domain shift analysis
    # ============================================================
    logger.info("=" * 60)
    logger.info("DOMAIN SHIFT ANALYSIS")
    logger.info("=" * 60)

    X_val_flat = X_val.reshape(len(X_val), -1)[:, :pipeline.n_features_final]

    try:
        X_cse_benign_raw, y_cse_all, meta_cse = load_all_labeled(
            cse_files, pipeline.feature_names, cse_label, session_cfg,
            column_mapper=cse_mapper,
            chunksize=pp_cfg.get("chunksize", 50000),
        )
        cse_benign_mask = binary_labels(y_cse_all, benign_label) == 0
        X_cse_benign_scaled = pipeline.transform(
            X_cse_benign_raw[cse_benign_mask].head(50000), reshape_2d=False,
        )

        ks_df = ks_test_per_feature(
            X_val_flat[:50000], X_cse_benign_scaled[:50000],
            pipeline.feature_names,
        )
        plot_domain_shift(ks_df, save_path=str(Path(results_dir) / "domain_shift.png"))
        shift_summary = summarize_shift(ks_df)
        report["domain_shift"] = shift_summary
        save_json(Path(results_dir) / "domain_shift.json", shift_summary)
        ks_df.to_csv(Path(results_dir) / "domain_shift_features.csv", index=False)
    except Exception as e:
        logger.warning("Domain shift analysis failed: %s", e)
        report["domain_shift"] = {"error": str(e)}

    # ============================================================
    # 5. Train Stage 1 MSCNN-AE
    # ============================================================
    logger.info("=" * 60)
    logger.info("STAGE 1: MSCNN-AE TRAINING")
    logger.info("=" * 60)

    s1_model, s1_encoder, s1_history = train_stage1(
        X_train, X_val, cfg, models_dir, results_dir,
    )

    report["stage1"] = {
        "final_train_loss": float(s1_history.history["loss"][-1]),
        "final_val_loss": float(s1_history.history["val_loss"][-1]),
        "best_val_loss": float(min(s1_history.history["val_loss"])),
        "n_epochs": len(s1_history.history["loss"]),
        "total_params": s1_model.count_params(),
    }

    # Extract latent vectors
    latent_train = extract_latent_vectors(s1_encoder, X_train)
    latent_val = extract_latent_vectors(s1_encoder, X_val)

    err_s1_train = compute_stage1_errors(s1_model, X_train)
    err_s1_val = compute_stage1_errors(s1_model, X_val)

    save_npz(
        Path(processed_dir) / "latent_train.npz",
        latent=latent_train, errors=err_s1_train,
    )
    save_npz(
        Path(processed_dir) / "latent_val.npz",
        latent=latent_val, errors=err_s1_val,
    )

    # ============================================================
    # 6. Session analysis & windowing
    # ============================================================
    logger.info("=" * 60)
    logger.info("SESSION ANALYSIS & WINDOWING")
    logger.info("=" * 60)

    session_stats_train = analyze_session_lengths(meta_train)
    report["session_stats"] = session_stats_train

    plot_session_lengths(
        meta_train, save_path=str(Path(results_dir) / "session_lengths.png"),
    )

    windows_train, _, eff_W = create_windows(
        latent_train, meta_train, session_stats_train, window_cfg,
    )
    windows_val, _, _ = create_windows(
        latent_val, meta_val, session_stats_train, window_cfg,
    )

    report["effective_window_size"] = eff_W
    logger.info("Effective window size: W=%d", eff_W)

    # ============================================================
    # 7. Train Stage 2 BiLSTM-AE
    # ============================================================
    logger.info("=" * 60)
    logger.info("STAGE 2: BiLSTM/DENSE-AE TRAINING")
    logger.info("=" * 60)

    s2_model, s2_encoder, s2_history = train_stage2(
        windows_train, windows_val,
        latent_dim=pipeline.latent_dim,
        window_size=eff_W,
        cfg=cfg,
        models_dir=models_dir,
        results_dir=results_dir,
    )

    report["stage2"] = {
        "model_type": "BiLSTM-AE" if eff_W > 1 else "Dense-AE",
        "final_train_loss": float(s2_history.history["loss"][-1]),
        "final_val_loss": float(s2_history.history["val_loss"][-1]),
        "best_val_loss": float(min(s2_history.history["val_loss"])),
        "n_epochs": len(s2_history.history["loss"]),
        "total_params": s2_model.count_params(),
    }

    # ============================================================
    # 8. Threshold determination (benign val only)
    # ============================================================
    logger.info("=" * 60)
    logger.info("THRESHOLD DETERMINATION")
    logger.info("=" * 60)

    err_s2_val = compute_stage2_errors(s2_model, windows_val)

    alpha = scoring_cfg.get("alpha", 0.5)

    if eff_W > 1:
        n_flows_covered = len(err_s2_val) * eff_W
        if n_flows_covered < len(err_s1_val):
            err_s1_val_for_combine = err_s1_val[:n_flows_covered]
        else:
            err_s1_val_for_combine = err_s1_val

        s1_per_window = err_s1_val_for_combine[:len(err_s2_val) * eff_W].reshape(-1, eff_W).mean(axis=1)
        benign_combined = combine_scores(s1_per_window, err_s2_val, alpha=alpha)
    else:
        benign_combined = combine_scores(err_s1_val, err_s2_val, alpha=alpha)

    threshold_results = compute_all_thresholds(benign_combined, cfg)
    report["thresholds"] = threshold_results
    save_json(Path(results_dir) / "thresholds.json", threshold_results)

    selected_threshold = threshold_results["selected_threshold"]
    logger.info("Selected threshold: %.6f", selected_threshold)

    # ============================================================
    # 9. Evaluate on CIC-2017 all-label
    # ============================================================
    logger.info("=" * 60)
    logger.info("EVALUATION: CIC-IDS-2017 (ALL LABELS)")
    logger.info("=" * 60)

    cic_metrics, cic_curves = _evaluate_dataset(
        cic_files, pipeline, s1_model, s1_encoder, s2_model,
        cic_label, session_cfg, session_stats_train, window_cfg,
        eff_W, alpha, selected_threshold, benign_label,
        dataset_name="CIC-2017", results_dir=results_dir,
        column_mapper=None, chunksize=pp_cfg.get("chunksize", 50000),
    )
    report["cic_metrics"] = cic_metrics

    # ============================================================
    # 10. Evaluate on CSE-CIC-IDS-2018 (PRIMARY)
    # ============================================================
    logger.info("=" * 60)
    logger.info("EVALUATION: CSE-CIC-IDS-2018 (PRIMARY — UNSEEN)")
    logger.info("=" * 60)

    cse_metrics, cse_curves = _evaluate_dataset(
        cse_files, pipeline, s1_model, s1_encoder, s2_model,
        cse_label, session_cfg, session_stats_train, window_cfg,
        eff_W, alpha, selected_threshold, benign_label,
        dataset_name="CSE-2018", results_dir=results_dir,
        column_mapper=cse_mapper, chunksize=pp_cfg.get("chunksize", 50000),
    )
    report["cse_metrics"] = cse_metrics

    # ============================================================
    # 11. Combined visualizations
    # ============================================================
    logger.info("Generating combined visualizations...")

    plot_roc_curves(
        {"CIC-2017": cic_curves["roc"], "CSE-2018": cse_curves["roc"]},
        save_path=str(Path(results_dir) / "roc_curves_combined.png"),
    )
    plot_pr_curves(
        {"CIC-2017": cic_curves["pr"], "CSE-2018": cse_curves["pr"]},
        save_path=str(Path(results_dir) / "pr_curves_combined.png"),
    )

    # Threshold comparison on CSE-2018
    if "y_true" in cse_curves and "scores" in cse_curves:
        comp_df = build_threshold_comparison_table(
            threshold_results, cse_curves["y_true"], cse_curves["scores"],
            dataset_name="CSE-2018",
        )
        comp_df.to_csv(Path(results_dir) / "threshold_comparison.csv", index=False)
        plot_threshold_comparison(
            comp_df,
            save_path=str(Path(results_dir) / "threshold_comparison.png"),
        )

    # ============================================================
    # 12. Generalization Analysis
    # ============================================================
    logger.info("=" * 60)
    logger.info("GENERALIZATION ANALYSIS")
    logger.info("=" * 60)

    gen_analysis = {
        "cic_roc_auc": cic_metrics.get("roc_auc", 0),
        "cse_roc_auc": cse_metrics.get("roc_auc", 0),
        "cic_pr_auc": cic_metrics.get("pr_auc", 0),
        "cse_pr_auc": cse_metrics.get("pr_auc", 0),
        "cic_f1": cic_metrics.get("f1", 0),
        "cse_f1": cse_metrics.get("f1", 0),
        "auc_drop": cic_metrics.get("roc_auc", 0) - cse_metrics.get("roc_auc", 0),
        "f1_drop": cic_metrics.get("f1", 0) - cse_metrics.get("f1", 0),
    }

    if gen_analysis["auc_drop"] < 0.10:
        gen_analysis["verdict"] = "GOOD generalization"
    elif gen_analysis["auc_drop"] < 0.20:
        gen_analysis["verdict"] = "MODERATE generalization"
    else:
        gen_analysis["verdict"] = "POOR generalization — likely overfitting to CIC-2017"

    report["generalization"] = gen_analysis
    logger.info("Generalization verdict: %s", gen_analysis["verdict"])
    logger.info(
        "AUC drop: %.4f, F1 drop: %.4f",
        gen_analysis["auc_drop"], gen_analysis["f1_drop"],
    )

    # Save final report
    elapsed = time.time() - t0
    report["elapsed_seconds"] = elapsed
    save_json(Path(results_dir) / "final_report.json", report)
    logger.info("Pipeline complete in %.1f seconds. Report saved.", elapsed)

    return report


def _evaluate_dataset(
    csv_files, pipeline, s1_model, s1_encoder, s2_model,
    label_col, session_cfg, session_stats, window_cfg,
    eff_W, alpha, threshold, benign_label,
    dataset_name, results_dir,
    column_mapper=None, chunksize=50000,
) -> tuple[dict, dict]:
    """Evaluate a full dataset through both stages."""
    from src.data.preprocessing import load_all_labeled
    from src.data.windowing import create_windows
    from src.training.trainer import (
        compute_stage1_errors,
        compute_stage2_errors,
        extract_latent_vectors,
    )
    from src.training.threshold import combine_scores
    from src.evaluation.metrics import (
        binary_labels,
        compute_metrics,
        compute_pr_curve,
        compute_roc_curve,
        per_attack_detection_rate,
    )
    from src.evaluation.visualization import (
        plot_confusion_matrix,
        plot_error_distribution,
        plot_error_violin,
        plot_per_attack_dr,
    )

    X_raw, y_str, meta = load_all_labeled(
        csv_files, pipeline.feature_names, label_col, session_cfg,
        column_mapper=column_mapper, chunksize=chunksize,
    )

    X_2d = pipeline.transform(X_raw, reshape_2d=True)
    y_bin = binary_labels(y_str, benign_label)

    err_s1 = compute_stage1_errors(s1_model, X_2d)
    latent = extract_latent_vectors(s1_encoder, X_2d)

    windows, window_labels, _ = create_windows(
        latent, meta, session_stats, window_cfg, labels=y_bin,
    )

    err_s2 = compute_stage2_errors(s2_model, windows)

    if eff_W > 1 and window_labels is not None:
        n_windows = len(err_s2)
        s1_per_window = err_s1[:n_windows * eff_W].reshape(-1, eff_W).mean(axis=1)
        scores = combine_scores(s1_per_window, err_s2, alpha=alpha)
        eval_labels = window_labels
        eval_labels_str = np.where(window_labels == 0, "BENIGN", "ATTACK")
    else:
        scores = combine_scores(err_s1, err_s2, alpha=alpha)
        eval_labels = y_bin
        eval_labels_str = y_str.values if hasattr(y_str, 'values') else y_str

    metrics = compute_metrics(eval_labels, scores, threshold, dataset_name)

    roc = compute_roc_curve(eval_labels, scores)
    pr = compute_pr_curve(eval_labels, scores)

    benign_scores = scores[eval_labels == 0]
    attack_scores = scores[eval_labels == 1]
    plot_error_distribution(
        benign_scores, attack_scores, threshold,
        title=f"{dataset_name} — Error Distribution",
        save_path=str(Path(results_dir) / f"{dataset_name.lower().replace('-','')}_error_dist.png"),
    )

    plot_error_violin(
        scores, eval_labels_str, threshold,
        title=f"{dataset_name} — Error by Category",
        save_path=str(Path(results_dir) / f"{dataset_name.lower().replace('-','')}_violin.png"),
    )

    plot_confusion_matrix(
        metrics["confusion_matrix"],
        title=f"{dataset_name} — Confusion Matrix",
        save_path=str(Path(results_dir) / f"{dataset_name.lower().replace('-','')}_cm.png"),
    )

    dr_df = per_attack_detection_rate(eval_labels_str, scores, threshold, benign_label)
    dr_df.to_csv(Path(results_dir) / f"{dataset_name.lower().replace('-','')}_detection_rates.csv", index=False)
    plot_per_attack_dr(
        dr_df, title=f"{dataset_name} — Detection Rate",
        save_path=str(Path(results_dir) / f"{dataset_name.lower().replace('-','')}_dr.png"),
    )

    metrics["per_attack_dr"] = dr_df.to_dict("records")
    save_json(Path(results_dir) / f"{dataset_name.lower().replace('-','')}_metrics.json", metrics)

    curves = {"roc": roc, "pr": pr, "y_true": eval_labels, "scores": scores}
    return metrics, curves


def main():
    parser = argparse.ArgumentParser(description="MSCNN-BiLSTM-AE NIDS Pipeline")
    parser.add_argument(
        "--config", type=str, default="src/config.yaml",
        help="Path to config YAML file",
    )
    args = parser.parse_args()

    setup_logging("INFO")
    cfg = load_yaml(args.config)
    report = run_pipeline(cfg)

    logger.info("=" * 60)
    logger.info("FINAL RESULTS SUMMARY")
    logger.info("=" * 60)
    logger.info("CIC-2017 ROC-AUC: %.4f", report.get("cic_metrics", {}).get("roc_auc", 0))
    logger.info("CSE-2018 ROC-AUC: %.4f", report.get("cse_metrics", {}).get("roc_auc", 0))
    logger.info("CIC-2017 F1:      %.4f", report.get("cic_metrics", {}).get("f1", 0))
    logger.info("CSE-2018 F1:      %.4f", report.get("cse_metrics", {}).get("f1", 0))
    logger.info("Generalization:   %s", report.get("generalization", {}).get("verdict", "N/A"))


if __name__ == "__main__":
    main()
