from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.audio import ensure_mono
from common.export import save_multi_sheet_excel


LABEL_MAP = {
    (0, 0): "normal",
    (1, 0): "crackles",
    (0, 1): "wheezes",
    (1, 1): "both",
}


# SoundProcessing/ICBHI_final_database
DATASET_DIR = PROJECT_ROOT / "ICBHI_final_database"

# SoundProcessing/segments_wav/annotation
SEGMENTS_DIR = PROJECT_ROOT / "segments_wav" / "annotation"

# SoundProcessing/segmentation/annotation
CURRENT_DIR = Path(__file__).resolve().parent

CSV_OUTPUT = CURRENT_DIR / "annotation_segmentation.csv"
XLSX_OUTPUT = CURRENT_DIR / "annotation_segmentation.xlsx"


def read_annotation(txt_path: Path) -> list[tuple[float, float, int, int]]:

    rows = []

    with txt_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 4:
                print(f"[WARN] Line {line_number} invalid in {txt_path.name}: {line}")
                continue

            try:
                start_s = float(parts[0])
                end_s = float(parts[1])
                crackles = int(parts[2])
                wheezes = int(parts[3])
            except ValueError:
                print(f"[WARN] Couldn't convert values {txt_path.name}, line {line_number}: {line}")
                continue

            rows.append((start_s, end_s, crackles, wheezes))

    return rows


def safe_slice(x: np.ndarray, sr: int, start_s: float, end_s: float) -> np.ndarray:

    start_i = max(0, int(round(start_s * sr)))
    end_i = min(len(x), int(round(end_s * sr)))

    if end_i <= start_i:
        return np.array([], dtype=np.float32)

    return x[start_i:end_i]


def get_label(crackles: int, wheezes: int) -> str:
    return LABEL_MAP.get((crackles, wheezes), "unknown")


def segment_file(wav_path: Path, txt_path: Path, segments_dir: Path) -> list[dict]:

    try:
        x, sr = sf.read(wav_path)
    except Exception as e:
        print(f"[ERROR] Could not read {wav_path.name}: {e}")
        return []

    x = ensure_mono(x, dtype=np.float32)
    annotations = read_annotation(txt_path)

    rows = []

    for idx, (start_s, end_s, crackles, wheezes) in enumerate(annotations):
        segment = safe_slice(x, sr, start_s, end_s)
        label = get_label(crackles, wheezes)
        duration_s = len(segment) / sr

        segment_filename = f"{wav_path.stem}_seg_{idx:04d}_{label}.wav"
        segment_path = segments_dir / segment_filename

        try:
            sf.write(segment_path, segment, sr)
        except Exception as e:
            print(f"[ERROR] Could not save {segment_filename}: {e}")
            continue

        rows.append({
            "original_file": wav_path.name,
            "annotation_file": txt_path.name,
            "segment_file": segment_filename,
            "segment_path": str(segment_path.relative_to(PROJECT_ROOT)),
            "segment_id": idx,
            "start_s": start_s,
            "end_s": end_s,
            "duration_s": duration_s,
            "label": label,
            "crackles": crackles,
            "wheezes": wheezes,
            "sample_rate": sr,
            "num_samples": len(segment),
        })

    return rows


def process_dataset(dataset_dir: Path, segments_dir: Path) -> list[dict]:

    all_rows = []

    wav_files = sorted(dataset_dir.glob("*.wav"))
    print(f"[INFO] Audio files {len(wav_files)} found in {dataset_dir}")

    for wav_path in wav_files:
        txt_path = dataset_dir / f"{wav_path.stem}.txt"

        if not txt_path.exists():
            print(f"[WARN] Missing annotations for {wav_path.name}")
            continue

        file_rows = segment_file(wav_path, txt_path, segments_dir)
        print(f"[INFO] {wav_path.name}: {len(file_rows)} saved segments")
        all_rows.extend(file_rows)

    return all_rows


def save_outputs(rows: list[dict], csv_path: Path, xlsx_path: Path) -> None:

    columns = [
        "original_file",
        "annotation_file",
        "segment_file",
        "segment_path",
        "segment_id",
        "start_s",
        "end_s",
        "duration_s",
        "label",
        "crackles",
        "wheezes",
        "sample_rate",
        "num_samples",
    ]

    df = pd.DataFrame(rows, columns=columns)

    df.to_csv(csv_path, index=False)
    save_multi_sheet_excel([("segments", df)], xlsx_path)

    print(f"[INFO] CSV saved: {csv_path}")
    print(f"[INFO] XLSX saved: {xlsx_path}")
    print(f"[INFO] Shape tabel: {df.shape}")


def main() -> None:
    print(f"[INFO] PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"[INFO] DATASET_DIR: {DATASET_DIR}")
    print(f"[INFO] SEGMENTS_DIR: {SEGMENTS_DIR}")

    if not DATASET_DIR.exists():
        print(f"[ERROR] The dataset folder not found: {DATASET_DIR}")
        return

    SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = process_dataset(DATASET_DIR, SEGMENTS_DIR)

    if not rows:
        print("[WARN] No generated segments.")
        return

    save_outputs(rows, CSV_OUTPUT, XLSX_OUTPUT)

    print(f"[DONE] Total saved segments: {len(rows)}")
    print(f"[DONE] Audio files : {SEGMENTS_DIR}")


if __name__ == "__main__":
    main()
