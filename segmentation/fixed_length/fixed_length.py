from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import soundfile as sf

PROJECT_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_BOOTSTRAP))

from common.export import save_multi_sheet_excel
from common.paths import find_project_root


PROJECT_ROOT = find_project_root("SoundProcessing")
SCRIPT_DIR = Path(__file__).resolve().parent

AUDIO_DIR = PROJECT_ROOT / "ICBHI_final_database"
OUT_AUDIO_DIR = PROJECT_ROOT / "brutal_fixed_segments"

OUT_CSV = SCRIPT_DIR / "brutal_fixed_segmentation.csv"
OUT_XLSX = SCRIPT_DIR / "brutal_fixed_segmentation.xlsx"

OUT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

print("Project root:", PROJECT_ROOT)
print("Script dir:", SCRIPT_DIR)
print("Audio dir:", AUDIO_DIR)
print("Output segments dir:", OUT_AUDIO_DIR)
print("Output CSV:", OUT_CSV)
print("Output XLSX:", OUT_XLSX)


def brutal_fixed_length_segmentation(segment_length_sec: float = 2.5) -> None:
    if not AUDIO_DIR.exists():
        raise FileNotFoundError(f"Audio directory not found: {AUDIO_DIR}")

    wav_files = sorted(AUDIO_DIR.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No .wav files found in: {AUDIO_DIR}")

    print(f"Found {len(wav_files)} wav files.")

    segment_rows = []
    summary_rows = []
    brutal_segment_id = 1

    for wav_path in wav_files:
        signal, sr = sf.read(wav_path)


        if signal.ndim > 1:
            signal = np.mean(signal, axis=1)

        samples_per_segment = int(segment_length_sec * sr)
        total_samples = len(signal)
        total_duration_s = total_samples / sr
        num_segments = total_samples // samples_per_segment
        remainder_samples = total_samples % samples_per_segment
        remainder_duration_s = remainder_samples / sr

        print(f"\nProcessing: {wav_path.name}")
        print(f"  Sample rate: {sr}")
        print(f"  Total samples: {total_samples}")
        print(f"  Samples/segment: {samples_per_segment}")
        print(f"  Full segments: {num_segments}")
        print(f"  Remainder discarded: {remainder_samples} samples")


        summary_rows.append({
            "recording": wav_path.name,
            "sample_rate": sr,
            "total_samples": total_samples,
            "total_duration_s": round(total_duration_s, 3),
            "segment_length_s": segment_length_sec,
            "samples_per_segment": samples_per_segment,
            "full_segments": num_segments,
            "remainder_samples": remainder_samples,
            "remainder_duration_s": round(remainder_duration_s, 3)
        })


        for seg_idx in range(num_segments):
            start_sample = seg_idx * samples_per_segment
            end_sample = start_sample + samples_per_segment
            segment = signal[start_sample:end_sample]

            start_s = start_sample / sr
            end_s = end_sample / sr

            segment_filename = f"{wav_path.stem}_seg_{seg_idx + 1}.wav"
            out_path = OUT_AUDIO_DIR / segment_filename
            sf.write(out_path, segment, sr)

            segment_rows.append({
                "brutal_segment_id": brutal_segment_id,
                "recording": wav_path.name,
                "segment_in_recording": seg_idx + 1,
                "segment_filename": segment_filename,
                "start_sample": start_sample,
                "end_sample_excl": end_sample,
                "start_s": round(start_s, 3),
                "end_s": round(end_s, 3),
                "duration_s": round(end_s - start_s, 3),
                "sample_rate": sr,
                "samples_per_segment": samples_per_segment
            })

            brutal_segment_id += 1

    df_segments = pd.DataFrame(segment_rows)
    df_summary = pd.DataFrame(summary_rows)


    df_segments.to_csv(OUT_CSV, index=False)


    save_multi_sheet_excel(
        [("files_summary", df_summary), ("segments", df_segments)],
        OUT_XLSX,
    )

    print("\n[OK] Brutal fixed-length segmentation finished.")
    print("CSV saved to:", OUT_CSV.resolve())
    print("XLSX saved to:", OUT_XLSX.resolve())
    print(f"Total segments created: {len(df_segments)}")


if __name__ == "__main__":
    brutal_fixed_length_segmentation(segment_length_sec=2.5)
