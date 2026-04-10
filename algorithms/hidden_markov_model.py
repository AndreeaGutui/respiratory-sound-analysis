from __future__ import annotations

import random
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from hmmlearn.hmm import GaussianHMM
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler


def find_project_root(project_name: str = "SoundProcessing") -> Path:
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if parent.name == project_name:
            return parent
    raise RuntimeError(f"Project root '{project_name}' not found from {current}")


def label_to_binary(label: str) -> int:
    return 0 if str(label).strip().lower() == "normal" else 1


def save_excel(df: pd.DataFrame, out_xlsx: Path, sheet_name: str = "results") -> None:
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)


def load_annotation_reference(annotation_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(annotation_csv)
    df["binary_label"] = df["label"].apply(label_to_binary)
    return df


def parse_seg_filename(filename: str) -> tuple[str, int] | None:
    match = re.match(r"^(?P<recording>.+)_seg_(?P<index>\d+)\.wav$", filename)
    if not match:
        return None
    return match.group("recording"), int(match.group("index"))


def assign_binary_label_by_overlap(recording: str, start_s: float, end_s: float, annotation_df: pd.DataFrame) -> int | None:
    recording_rows = annotation_df[annotation_df["original_file"] == recording]
    if recording_rows.empty:
        return None

    best_overlap = 0.0
    best_label = None

    for row in recording_rows.itertuples(index=False):
        overlap = max(0.0, min(end_s, row.end_s) - max(start_s, row.start_s))
        if overlap > best_overlap:
            best_overlap = overlap
            best_label = int(row.binary_label)

    if best_overlap <= 0 or best_label is None:
        return None
    return best_label


def build_metadata_lookup(method_name: str, project_root: Path, annotation_df: pd.DataFrame) -> dict[object, int]:
    if method_name == "labeled":
        return {}

    if method_name == "fixed_length":
        metadata_csv = project_root / "segmentation" / "fixed_length" / "brutal_fixed_segmentation.csv"
        metadata_df = pd.read_csv(metadata_csv)
        name_column = "segment_filename"
    elif method_name == "max_spectral_centroid":
        metadata_csv = project_root / "segmentation" / "max_spectral_centroid" / "max_spectral_centroid_segmentation.csv"
        metadata_df = pd.read_csv(metadata_csv)
        name_column = "cycle_filename"
    elif method_name == "spectral_centroid_slope":
        metadata_csv = project_root / "segmentation" / "spectral_centroid_slope" / "spectral_centroid_slope_segmentation.csv"
        metadata_df = pd.read_csv(metadata_csv)
        name_column = "cycle_filename"
    else:
        raise ValueError(f"Unknown method: {method_name}")

    lookup: dict[object, int] = {}

    for row in metadata_df.itertuples(index=False):
        label = assign_binary_label_by_overlap(row.recording, float(row.start_s), float(row.end_s), annotation_df)
        if label is not None:
            lookup[getattr(row, name_column)] = label

    if method_name in {"max_spectral_centroid", "spectral_centroid_slope"}:
        for row in metadata_df.itertuples(index=False):
            recording_key = str(row.recording).replace(".wav", "")
            if hasattr(row, "cycle_in_recording"):
                label = assign_binary_label_by_overlap(row.recording, float(row.start_s), float(row.end_s), annotation_df)
                if label is not None:
                    lookup[(recording_key, int(row.cycle_in_recording))] = label

    return lookup


def get_label_for_row(method_name: str, row: pd.Series, metadata_lookup: dict[object, int]) -> int | None:
    if method_name == "labeled":
        if pd.isna(row.get("label")):
            return None
        return label_to_binary(str(row["label"]))

    file_name = str(row["file_name"])
    if file_name in metadata_lookup:
        return int(metadata_lookup[file_name])

    parsed = parse_seg_filename(file_name)
    if parsed is not None and parsed in metadata_lookup:
        return int(metadata_lookup[parsed])

    return None


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {
        "segmentation_type",
        "file_name",
        "sample_rate",
        "n_samples",
        "duration_sec",
        "patient_id",
        "recording_id",
        "segment_id",
        "label",
        "binary_label",
    }
    return [column for column in df.columns if column not in excluded]


def split_list(items: list, train_ratio: float = 0.6, val_ratio: float = 0.2, test_ratio: float = 0.2) -> tuple[list, list, list]:
    n = len(items)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_items = items[:n_train]
    val_items = items[n_train:n_train + n_val]
    test_items = items[n_train + n_val:]
    return train_items, val_items, test_items


def plot_confusion_matrix(tn: int, fp: int, fn: int, tp: int, method_name: str, out_dir: Path) -> None:
    cm = np.array([[tn, fp], [fn, tp]])
    labels = np.array(
        [
            [f"TN\n{tn}", f"FP\n{fp}"],
            [f"FN\n{fn}", f"TP\n{tp}"],
        ]
    )

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=labels,
        fmt="",
        cmap="Blues",
        cbar=True,
        xticklabels=["Predicted Normal", "Predicted Anomaly"],
        yticklabels=["Actual Normal", "Actual Anomaly"],
        linewidths=1,
        linecolor="black",
    )
    plt.title(f"HMM Confusion Matrix - {method_name}")
    plt.tight_layout()
    plt.savefig(out_dir / f"hmm_confusion_matrix_{method_name}.png", dpi=300, bbox_inches="tight")
    plt.close()


def evaluate_model(
    model: GaussianHMM,
    normal_test: list[tuple[str, np.ndarray]],
    anomaly_test: list[tuple[str, np.ndarray]],
    threshold: float,
    method_name: str,
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    y_true = []
    y_pred = []
    rows = []

    for name, seq in normal_test:
        score = float(model.score(seq))
        pred = 0 if score >= threshold else 1
        y_true.append(0)
        y_pred.append(pred)
        rows.append(
            {
                "method_name": method_name,
                "segment_filename": name,
                "true_label": 0,
                "predicted_label": pred,
                "score": score,
                "threshold_log_likelihood": threshold,
            }
        )

    for name, seq in anomaly_test:
        score = float(model.score(seq))
        pred = 0 if score >= threshold else 1
        y_true.append(1)
        y_pred.append(pred)
        rows.append(
            {
                "method_name": method_name,
                "segment_filename": name,
                "true_label": 1,
                "predicted_label": pred,
                "score": score,
                "threshold_log_likelihood": threshold,
            }
        )

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_anomaly": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_anomaly": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_anomaly": float(f1_score(y_true, y_pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "classification_report": classification_report(y_true, y_pred, target_names=["normal", "anomaly"]),
    }
    return pd.DataFrame(rows), metrics


def process_method(method_name: str, feature_csv: Path, annotation_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"\n========== HMM: {method_name} ==========")
    print("Feature CSV:", feature_csv)

    df = pd.read_csv(feature_csv)
    metadata_lookup = build_metadata_lookup(method_name, PROJECT_ROOT, annotation_df)
    df["binary_label"] = df.apply(lambda row: get_label_for_row(method_name, row, metadata_lookup), axis=1)

    skipped = int(df["binary_label"].isna().sum())
    if skipped:
        print("Rows skipped because label was not found:", skipped)

    df = df.dropna(subset=["binary_label"]).copy()
    df["binary_label"] = df["binary_label"].astype(int)

    feature_cols = get_feature_columns(df)
    normal_segments: list[tuple[str, np.ndarray]] = []
    anomaly_segments: list[tuple[str, np.ndarray]] = []

    for _, row in df.iterrows():
        seq = row[feature_cols].to_numpy(dtype=np.float64).reshape(1, -1)
        item = (str(row["file_name"]), seq)
        if int(row["binary_label"]) == 0:
            normal_segments.append(item)
        else:
            anomaly_segments.append(item)

    print("Normal segments:", len(normal_segments))
    print("Anomaly segments:", len(anomaly_segments))

    rng = random.Random(42)
    rng.shuffle(normal_segments)
    rng.shuffle(anomaly_segments)

    normal_train, normal_val, normal_test = split_list(normal_segments, 0.6, 0.2, 0.2)
    _, _, anomaly_test = split_list(anomaly_segments, 0.0, 0.0, 1.0)

    print("Normal train:", len(normal_train))
    print("Normal val:", len(normal_val))
    print("Normal test:", len(normal_test))
    print("Anomaly test:", len(anomaly_test))

    # Train only on normal segments, because we use HMM like an anomaly detector.
    X_train_raw = np.vstack([seq for _, seq in normal_train])
    scaler = StandardScaler()
    scaler.fit(X_train_raw)

    normal_train_scaled = [(name, scaler.transform(seq)) for name, seq in normal_train]
    normal_val_scaled = [(name, scaler.transform(seq)) for name, seq in normal_val]
    normal_test_scaled = [(name, scaler.transform(seq)) for name, seq in normal_test]
    anomaly_test_scaled = [(name, scaler.transform(seq)) for name, seq in anomaly_test]

    X_train = np.vstack([seq for _, seq in normal_train_scaled])
    lengths_train = [len(seq) for _, seq in normal_train_scaled]

    # Each CSV row is one segment summary, so we keep a 1-state HMM.
    model = GaussianHMM(
        n_components=1,
        covariance_type="diag",
        n_iter=100,
        random_state=42,
        min_covar=1e-3,
        init_params="mc",
        params="mc",
    )
    model.startprob_ = np.array([1.0], dtype=np.float64)
    model.transmat_ = np.array([[1.0]], dtype=np.float64)
    model.fit(X_train, lengths_train)

    # Validation scores from normal data only. Low score means likely anomaly.
    scores_val = [float(model.score(seq)) for _, seq in normal_val_scaled]
    threshold = float(np.percentile(scores_val, 15))
    print("Threshold (log-likelihood):", threshold)

    predictions_df, metrics = evaluate_model(
        model=model,
        normal_test=normal_test_scaled,
        anomaly_test=anomaly_test_scaled,
        threshold=threshold,
        method_name=method_name,
    )

    print(metrics["classification_report"])
    print("Confusion matrix:")
    print(np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]]))

    plot_confusion_matrix(metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"], method_name, OUT_DIR)

    summary = pd.DataFrame(
        [
            {
                "method_name": method_name,
                "total_segments": len(normal_segments) + len(anomaly_segments),
                "normal_segments": len(normal_segments),
                "anomaly_segments": len(anomaly_segments),
                "normal_train": len(normal_train),
                "normal_val": len(normal_val),
                "normal_test": len(normal_test),
                "anomaly_test": len(anomaly_test),
                "threshold_log_likelihood": threshold,
                "accuracy": metrics["accuracy"],
                "precision_anomaly": metrics["precision_anomaly"],
                "recall_anomaly": metrics["recall_anomaly"],
                "f1_anomaly": metrics["f1_anomaly"],
                "tn": metrics["tn"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "tp": metrics["tp"],
            }
        ]
    )

    return summary, predictions_df


PROJECT_ROOT = find_project_root("SoundProcessing")
OUT_DIR = PROJECT_ROOT / "algorithms"
FEATURES_ROOT = PROJECT_ROOT / "features_csv"

METHODS = [
    ("labeled", FEATURES_ROOT / "labeled" / "labeled_features.csv"),
    ("fixed_length", FEATURES_ROOT / "fixed_length" / "fixed_length_features.csv"),
    ("max_spectral_centroid", FEATURES_ROOT / "max_spectral_centroid" / "max_spectral_centroid_features.csv"),
    ("spectral_centroid_slope", FEATURES_ROOT / "spectral_centroid_slope" / "spectral_centroid_slope_features.csv"),
]

ANNOTATION_CSV = PROJECT_ROOT / "segmentation" / "annotation" / "annotation_segmentation.csv"
COMPARISON_CSV = OUT_DIR / "hmm_segmentation_comparison_fixed.csv"
COMPARISON_XLSX = OUT_DIR / "hmm_segmentation_comparison_fixed.xlsx"
PREDICTIONS_CSV = OUT_DIR / "hmm_segmentation_predictions_fixed.csv"


def main() -> None:
    print("Project root:", PROJECT_ROOT)
    print("Output dir:", OUT_DIR)
    print("Annotation reference:", ANNOTATION_CSV)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    annotation_df = load_annotation_reference(ANNOTATION_CSV)

    comparison_rows = []
    prediction_rows = []

    for method_name, feature_csv in METHODS:
        comparison_df, predictions_df = process_method(method_name, feature_csv, annotation_df)
        comparison_rows.append(comparison_df)
        prediction_rows.append(predictions_df)

    comparison_all = pd.concat(comparison_rows, ignore_index=True).sort_values(by="f1_anomaly", ascending=False)
    predictions_all = pd.concat(prediction_rows, ignore_index=True)

    comparison_all.to_csv(COMPARISON_CSV, index=False)
    predictions_all.to_csv(PREDICTIONS_CSV, index=False)
    save_excel(comparison_all, COMPARISON_XLSX, sheet_name="comparison")

    print("\n[OK] HMM finished for all segmentation methods.")
    print("Comparison CSV:", COMPARISON_CSV.resolve())
    print("Comparison XLSX:", COMPARISON_XLSX.resolve())
    print("Predictions CSV:", PREDICTIONS_CSV.resolve())
    print("\nComparison summary:")
    print(comparison_all.to_string(index=False))


if __name__ == "__main__":
    main()
