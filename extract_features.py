from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import pandas as pd


SEGMENTATION_FOLDERS = (
    "fixed_length",
    "labeled",
    "max_spectral_centroid",
    "spectral_centroid_slope",
)


def summarize_feature(name: str, values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float32).ravel()
    if values.size == 0:
        return {f"{name}_mean": np.nan, f"{name}_std": np.nan}
    return {
        f"{name}_mean": float(np.mean(values)),
        f"{name}_std": float(np.std(values)),
    }


def parse_filename_metadata(wav_path: Path, segmentation_type: str) -> dict[str, str | int | None]:
    stem = wav_path.stem
    label = None
    segment_id = None

    if segmentation_type == "labeled":
        label_candidates = ("normal", "crackle", "wheeze", "both")
        for candidate in label_candidates:
            suffix = f"_{candidate}"
            if stem.endswith(suffix):
                label = candidate
                stem = stem[: -len(suffix)]
                break

    if "_seg_" in stem:
        base_name, segment_part = stem.rsplit("_seg_", 1)
        try:
            segment_id = int(segment_part)
        except ValueError:
            segment_id = segment_part
    else:
        base_name = stem

    patient_id = base_name.split("_", 1)[0] if "_" in base_name else base_name

    return {
        "patient_id": patient_id,
        "recording_id": base_name,
        "segment_id": segment_id,
        "label": label,
    }


def extract_features_from_file(wav_path: Path, segmentation_type: str, n_mfcc: int = 13) -> dict[str, str | int | float | None]:
    y, sr = librosa.load(wav_path, sr=None, mono=True)

    if y.size == 0:
        raise ValueError(f"Empty audio file: {wav_path}")

    features: dict[str, str | int | float | None] = {
        "segmentation_type": segmentation_type,
        "file_name": wav_path.name,
        "sample_rate": int(sr),
        "n_samples": int(len(y)),
        "duration_sec": float(len(y) / sr),
    }
    features.update(parse_filename_metadata(wav_path, segmentation_type))

    zcr = librosa.feature.zero_crossing_rate(y=y)
    rms = librosa.feature.rms(y=y)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    flatness = librosa.feature.spectral_flatness(y=y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    features.update(summarize_feature("zcr", zcr))
    features.update(summarize_feature("rms", rms))
    features.update(summarize_feature("spectral_centroid", centroid))
    features.update(summarize_feature("spectral_bandwidth", bandwidth))
    features.update(summarize_feature("spectral_rolloff", rolloff))
    features.update(summarize_feature("spectral_flatness", flatness))

    for index in range(n_mfcc):
        features.update(summarize_feature(f"mfcc_{index + 1}", mfcc[index]))

    return features


def extract_folder_features(
    input_dir: Path,
    output_dir: Path,
    segmentation_type: str,
    limit_files: int | None = None,
) -> Path:
    wav_files = sorted(input_dir.glob("*.wav"))
    if limit_files is not None:
        wav_files = wav_files[:limit_files]

    if not wav_files:
        raise FileNotFoundError(f"No .wav files found in {input_dir}")

    rows: list[dict[str, str | int | float | None]] = []
    for index, wav_path in enumerate(wav_files, start=1):
        rows.append(extract_features_from_file(wav_path, segmentation_type))
        if index % 100 == 0 or index == len(wav_files):
            print(f"[{segmentation_type}] {index}/{len(wav_files)} processed")

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{segmentation_type}_features.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract features from all segmentation folders and save them as CSV files."
    )
    parser.add_argument(
        "--segments-root",
        default="segments_wav",
        help="Root folder that contains the four segmentation subfolders.",
    )
    parser.add_argument(
        "--output-root",
        default="features_csv",
        help="Root folder where feature CSV subfolders will be created.",
    )
    parser.add_argument(
        "--limit-files",
        type=int,
        default=None,
        help="Optional limit for quick testing.",
    )
    args = parser.parse_args()

    segments_root = Path(args.segments_root)
    output_root = Path(args.output_root)

    if not segments_root.exists():
        raise FileNotFoundError(f"Segments root does not exist: {segments_root}")

    saved_paths: list[Path] = []
    for segmentation_type in SEGMENTATION_FOLDERS:
        input_dir = segments_root / segmentation_type
        output_dir = output_root / segmentation_type
        saved_csv = extract_folder_features(
            input_dir=input_dir,
            output_dir=output_dir,
            segmentation_type=segmentation_type,
            limit_files=args.limit_files,
        )
        saved_paths.append(saved_csv)

    print("\nFinished feature extraction.")
    for csv_path in saved_paths:
        print(csv_path.resolve())


if __name__ == "__main__":
    main()
