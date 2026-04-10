from __future__ import annotations
from pathlib import Path
import pandas as pd
import soundfile as sf
import numpy as np


def find_project_root(project_name: str = "SoundProcessing") -> Path:
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if parent.name == project_name:
            return parent
    raise RuntimeError(f"Project root '{project_name}' not found from {current}")


PROJECT_ROOT = find_project_root("SoundProcessing")

DATASET_DIR = PROJECT_ROOT / "ICBHI_final_database"
SEGMENTS_ROOT_DIR = PROJECT_ROOT / "segments_wav"

METHOD_TO_CSV = {
    "annotation": PROJECT_ROOT / "segmentation" / "annotation" / "annotation_segmentation.csv",
    "fixed_length": PROJECT_ROOT / "segmentation" / "fixed_length" / "brutal_fixed_segmentation.csv",
    "max_spectral_centroid": PROJECT_ROOT / "segmentation" / "max_spectral_centroid" / "max_spectral_centroid_segmentation.csv",
    "spectral_centroid_slope": PROJECT_ROOT / "segmentation" / "spectral_centroid_slope" / "spectral_centroid_slope_segmentation.csv",
}

print("Project root:", PROJECT_ROOT)
print("Dataset dir:", DATASET_DIR)
print("Segments root dir:", SEGMENTS_ROOT_DIR)


def find_audio_path(recording_value: str) -> Path:
    recording_stem = Path(str(recording_value)).stem
    wav_path = DATASET_DIR / f"{recording_stem}.wav"

    if not wav_path.exists():
        raise FileNotFoundError(f"Original audio file not found: {wav_path}")

    return wav_path


def choose_segment_index(row: pd.Series) -> int:
    for col in ["segment_in_recording", "cycle_in_recording", "segment_idx"]:
        if col in row and pd.notna(row[col]):
            return int(row[col])
    return 1


def export_segments_for_method(method_name: str, csv_path: Path) -> None:
    print(f"\n========== METHOD: {method_name} ==========")
    print("CSV:", csv_path)

    if not csv_path.exists():
        print(f"[SKIP] CSV not found: {csv_path}")
        return

    out_dir = SEGMENTS_ROOT_DIR / method_name
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    required_cols = {"recording", "start_s", "end_s"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"[SKIP] Missing required columns in {csv_path.name}: {missing}")
        return

    print(f"Rows in CSV: {len(df)}")

    exported = 0
    skipped = 0


    audio_cache: dict[str, tuple[np.ndarray, int]] = {}

    for i, row in df.iterrows():
        try:
            recording = str(row["recording"])
            start_s = float(row["start_s"])
            end_s = float(row["end_s"])

            if end_s <= start_s:
                print(f"  [SKIP row {i}] Invalid interval: start={start_s}, end={end_s}")
                skipped += 1
                continue

            wav_path = find_audio_path(recording)
            wav_key = str(wav_path)

            if wav_key not in audio_cache:
                signal, sr = sf.read(wav_path)

                if signal.ndim > 1:
                    signal = np.mean(signal, axis=1)

                signal = signal.astype(np.float64)
                audio_cache[wav_key] = (signal, sr)

            signal, sr = audio_cache[wav_key]

            total_duration_s = len(signal) / sr

            if start_s < 0 or end_s > total_duration_s:
                print(
                    f"  [SKIP row {i}] Out of bounds for {wav_path.name}: "
                    f"start={start_s:.3f}, end={end_s:.3f}, total={total_duration_s:.3f}"
                )
                skipped += 1
                continue

            start_sample = int(round(start_s * sr))
            end_sample = int(round(end_s * sr))

            if end_sample <= start_sample:
                print(f"  [SKIP row {i}] Empty segment after sample conversion.")
                skipped += 1
                continue

            segment = signal[start_sample:end_sample]

            seg_idx = choose_segment_index(row)

            out_name = f"{Path(recording).stem}_seg_{seg_idx}.wav"
            out_path = out_dir / out_name


            if out_path.exists():
                out_name = f"{Path(recording).stem}_seg_{seg_idx}_row_{i}.wav"
                out_path = out_dir / out_name

            sf.write(out_path, segment, sr)
            exported += 1

            if exported % 100 == 0:
                print(f"  Exported {exported} segments so far...")

        except Exception as e:
            print(f"  [SKIP row {i}] {e}")
            skipped += 1

    print(f"[OK] Method finished: {method_name}")
    print(f"Exported segments: {exported}")
    print(f"Skipped rows: {skipped}")
    print(f"Output folder: {out_dir.resolve()}")


def main() -> None:
    SEGMENTS_ROOT_DIR.mkdir(parents=True, exist_ok=True)

    for method_name, csv_path in METHOD_TO_CSV.items():
        export_segments_for_method(method_name, csv_path)

    print("\n[OK] All segment export operations finished.")


if __name__ == "__main__":
    main()