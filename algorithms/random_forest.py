from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


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


def save_table_image(df: pd.DataFrame, out_path: Path, title: str) -> None:
    fig_height = max(2.5, 0.6 * (len(df) + 1))
    fig, ax = plt.subplots(figsize=(14, fig_height))
    ax.axis("off")
    ax.set_title(title, fontsize=14, pad=14)

    show_df = df.copy()
    for col in show_df.select_dtypes(include=["float64", "float32"]).columns:
        show_df[col] = show_df[col].map(lambda x: f"{x:.4f}")

    table = ax.table(
        cellText=show_df.values,
        colLabels=show_df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.1, 1.4)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, method_name: str, out_dir: Path) -> None:
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        xticklabels=["Predicted Normal", "Predicted Anomaly"],
        yticklabels=["Actual Normal", "Actual Anomaly"],
        linewidths=1,
        linecolor="black",
    )
    plt.title(f"Random Forest Confusion Matrix - {method_name}")
    plt.tight_layout()
    plt.savefig(out_dir / f"random_forest_confusion_matrix_{method_name}.png", dpi=300, bbox_inches="tight")
    plt.close()


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
        index_column = "segment_in_recording"
    elif method_name == "max_spectral_centroid":
        metadata_csv = project_root / "segmentation" / "max_spectral_centroid" / "max_spectral_centroid_segmentation.csv"
        metadata_df = pd.read_csv(metadata_csv)
        name_column = "cycle_filename"
        index_column = "cycle_in_recording"
    elif method_name == "spectral_centroid_slope":
        metadata_csv = project_root / "segmentation" / "spectral_centroid_slope" / "spectral_centroid_slope_segmentation.csv"
        metadata_df = pd.read_csv(metadata_csv)
        name_column = "cycle_filename"
        index_column = "cycle_in_recording"
    else:
        raise ValueError(f"Unknown method: {method_name}")

    lookup: dict[object, int] = {}

    for row in metadata_df.itertuples(index=False):
        label = assign_binary_label_by_overlap(row.recording, float(row.start_s), float(row.end_s), annotation_df)
        if label is None:
            continue

        lookup[getattr(row, name_column)] = label
        recording_key = str(row.recording).replace(".wav", "")
        lookup[(recording_key, int(getattr(row, index_column)))] = label

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

    recording_id = row.get("recording_id")
    segment_id = row.get("segment_id")
    if pd.notna(recording_id) and pd.notna(segment_id):
        key = (str(recording_id), int(segment_id))
        if key in metadata_lookup:
            return int(metadata_lookup[key])

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


def process_method(method_name: str, feature_csv: Path, annotation_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print(f"\n========== RANDOM FOREST: {method_name} ==========")
    print("Feature CSV:", feature_csv)

    df = pd.read_csv(feature_csv)
    metadata_lookup = build_metadata_lookup(method_name, PROJECT_ROOT, annotation_df)
    df["binary_label"] = df.apply(lambda row: get_label_for_row(method_name, row, metadata_lookup), axis=1)

    skipped = int(df["binary_label"].isna().sum())
    if skipped:
        print("Rows skipped because label was not found:", skipped)

    df = df.dropna(subset=["binary_label"]).copy()
    df["binary_label"] = df["binary_label"].astype(int)

    print("Usable rows:", len(df))
    print("Normal:", int((df["binary_label"] == 0).sum()))
    print("Anomaly:", int((df["binary_label"] == 1).sum()))

    feature_cols = get_feature_columns(df)
    X = df[feature_cols]
    y = df["binary_label"]

    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X,
        y,
        df,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred)

    print(classification_report(y_test, y_pred, target_names=["normal", "anomaly"]))
    print("Confusion matrix:")
    print(cm)

    plot_confusion_matrix(cm, method_name, OUT_DIR)

    feature_importance_df = pd.DataFrame(
        {
            "method_name": method_name,
            "feature": feature_cols,
            "importance": model.feature_importances_,
        }
    ).sort_values(by="importance", ascending=False)

    results_df = df_test[["file_name", "recording_id", "segment_id", "binary_label"]].copy()
    results_df["method_name"] = method_name
    results_df["predicted_label"] = y_pred
    results_df["predicted_proba_anomaly"] = y_proba

    summary = pd.DataFrame(
        [
            {
                "method_name": method_name,
                "total_segments": len(df),
                "normal_segments": int((df["binary_label"] == 0).sum()),
                "anomaly_segments": int((df["binary_label"] == 1).sum()),
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision_anomaly": float(precision_score(y_test, y_pred, zero_division=0)),
                "recall_anomaly": float(recall_score(y_test, y_pred, zero_division=0)),
                "f1_anomaly": float(f1_score(y_test, y_pred, zero_division=0)),
                "tn": int(cm[0, 0]),
                "fp": int(cm[0, 1]),
                "fn": int(cm[1, 0]),
                "tp": int(cm[1, 1]),
            }
        ]
    )

    return summary, results_df, feature_importance_df


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
COMPARISON_CSV = OUT_DIR / "random_forest_segmentation_comparison.csv"
COMPARISON_XLSX = OUT_DIR / "random_forest_segmentation_comparison.xlsx"
COMPARISON_PNG = OUT_DIR / "random_forest_segmentation_comparison.png"
PREDICTIONS_CSV = OUT_DIR / "random_forest_segmentation_predictions.csv"
IMPORTANCES_CSV = OUT_DIR / "random_forest_feature_importances.csv"


def main() -> None:
    print("Project root:", PROJECT_ROOT)
    print("Output dir:", OUT_DIR)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    annotation_df = load_annotation_reference(ANNOTATION_CSV)

    all_summary = []
    all_predictions = []
    all_importances = []

    for method_name, feature_csv in METHODS:
        summary_df, predictions_df, importances_df = process_method(method_name, feature_csv, annotation_df)
        all_summary.append(summary_df)
        all_predictions.append(predictions_df)
        all_importances.append(importances_df)

    comparison_all = pd.concat(all_summary, ignore_index=True).sort_values(by="f1_anomaly", ascending=False)
    predictions_all = pd.concat(all_predictions, ignore_index=True)
    importances_all = pd.concat(all_importances, ignore_index=True)

    comparison_all.to_csv(COMPARISON_CSV, index=False)
    predictions_all.to_csv(PREDICTIONS_CSV, index=False)
    importances_all.to_csv(IMPORTANCES_CSV, index=False)
    save_excel(comparison_all, COMPARISON_XLSX, sheet_name="comparison")
    save_table_image(comparison_all, COMPARISON_PNG, "Random Forest Comparison Summary")

    print("\n[OK] Random Forest finished for all segmentation methods.")
    print("Comparison CSV:", COMPARISON_CSV.resolve())
    print("Comparison XLSX:", COMPARISON_XLSX.resolve())
    print("Comparison PNG:", COMPARISON_PNG.resolve())
    print("Predictions CSV:", PREDICTIONS_CSV.resolve())
    print("Feature importances CSV:", IMPORTANCES_CSV.resolve())
    print("\nComparison summary:")
    print(comparison_all.to_string(index=False))


if __name__ == "__main__":
    main()
