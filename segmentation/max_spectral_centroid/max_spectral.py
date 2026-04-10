from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import find_peaks

PROJECT_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_BOOTSTRAP))

from common.export import save_multi_sheet_excel
from common.paths import find_project_root
from common.spectral import spectral_centroid_per_frame


PROJECT_ROOT = find_project_root("SoundProcessing")
SCRIPT_DIR = Path(__file__).resolve().parent

AUDIO_DIR = PROJECT_ROOT / "ICBHI_final_database"
OUT_AUDIO_DIR = PROJECT_ROOT / "segmented_cycles"
OUT_CSV = SCRIPT_DIR / "max_spectral_centroid_segmentation.csv"
OUT_XLSX = SCRIPT_DIR / "max_spectral_centroid_segmentation.xlsx"

OUT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

print("Project root:", PROJECT_ROOT)
print("Script dir:", SCRIPT_DIR)
print("Audio dir:", AUDIO_DIR)
print("Output cycles dir:", OUT_AUDIO_DIR)
print("Output CSV:", OUT_CSV)
print("Output XLSX:", OUT_XLSX)
def max_spectral_centroid_segmentation(
    cycle_length_s: float = 2.5,
    peak_offset_before_s: float = 0.8,
    peak_offset_after_s: float = 1.7,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    min_peak_distance_s: float = 1.0,
    min_peak_percentile: float = 75.0
) -> None:
    if not AUDIO_DIR.exists():
        raise FileNotFoundError(f"Audio directory not found: {AUDIO_DIR}")

    wav_files = sorted(AUDIO_DIR.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No .wav files found in: {AUDIO_DIR}")

    print(f"Found {len(wav_files)} wav files.")

    cycle_rows = []
    summary_rows = []
    cycle_id = 1

    for wav_path in wav_files:
        signal, sr = sf.read(wav_path)

        if signal.ndim > 1:
            signal = np.mean(signal, axis=1)

        signal = signal.astype(np.float64)

        total_samples = len(signal)
        total_duration_s = total_samples / sr

        frame_len = int(sr * frame_ms / 1000.0)
        hop_len = int(sr * hop_ms / 1000.0)

        centroids = spectral_centroid_per_frame(signal, sr, frame_len, hop_len)

        if len(centroids) == 0:
            summary_rows.append({
                "recording": wav_path.name,
                "sample_rate": sr,
                "total_samples": total_samples,
                "total_duration_s": round(total_duration_s, 3),
                "n_frames": 0,
                "n_peaks_found": 0,
                "n_valid_cycles": 0,
                "n_discarded_boundary_peaks": 0,
                "centroid_threshold": None,
                "frame_ms": frame_ms,
                "hop_ms": hop_ms
            })
            continue

        threshold = np.percentile(centroids, min_peak_percentile)
        min_peak_distance_frames = max(1, int(min_peak_distance_s / (hop_len / sr)))

        peaks, properties = find_peaks(
            centroids,
            height=threshold,
            distance=min_peak_distance_frames
        )

        valid_cycles = 0
        boundary_discards = 0

        print(f"\nProcessing: {wav_path.name}")
        print(f"  Sample rate: {sr}")
        print(f"  Duration: {round(total_duration_s, 3)} s")
        print(f"  Frames: {len(centroids)}")
        print(f"  Threshold: {round(threshold, 2)}")
        print(f"  Peaks found: {len(peaks)}")

        for peak_idx, peak_frame in enumerate(peaks, start=1):
            peak_sample = peak_frame * hop_len
            peak_time_s = peak_sample / sr

            start_s = peak_time_s - peak_offset_before_s
            end_s = peak_time_s + peak_offset_after_s

            if start_s < 0 or end_s > total_duration_s:
                boundary_discards += 1
                continue

            start_sample = int(round(start_s * sr))
            end_sample = int(round(end_s * sr))

            cycle = signal[start_sample:end_sample]

            expected_len = int(round(cycle_length_s * sr))
            if len(cycle) != expected_len:
                # extra safety
                if len(cycle) < expected_len:
                    continue
                cycle = cycle[:expected_len]

            cycle_filename = f"{wav_path.stem}_cycle_{valid_cycles + 1}.wav"
            out_path = OUT_AUDIO_DIR / cycle_filename
            sf.write(out_path, cycle, sr)

            cycle_rows.append({
                "cycle_id": cycle_id,
                "recording": wav_path.name,
                "cycle_in_recording": valid_cycles + 1,
                "cycle_filename": cycle_filename,
                "peak_frame": int(peak_frame),
                "peak_time_s": round(peak_time_s, 3),
                "start_sample": start_sample,
                "end_sample_excl": end_sample,
                "start_s": round(start_s, 3),
                "end_s": round(end_s, 3),
                "duration_s": round((end_sample - start_sample) / sr, 3),
                "sample_rate": sr,
                "frame_ms": frame_ms,
                "hop_ms": hop_ms,
                "centroid_threshold": round(float(threshold), 2),
                "peak_centroid_value": round(float(centroids[peak_frame]), 2)
            })

            cycle_id += 1
            valid_cycles += 1

        summary_rows.append({
            "recording": wav_path.name,
            "sample_rate": sr,
            "total_samples": total_samples,
            "total_duration_s": round(total_duration_s, 3),
            "n_frames": len(centroids),
            "n_peaks_found": len(peaks),
            "n_valid_cycles": valid_cycles,
            "n_discarded_boundary_peaks": boundary_discards,
            "centroid_threshold": round(float(threshold), 2),
            "frame_ms": frame_ms,
            "hop_ms": hop_ms
        })

        print(f"  Valid cycles saved: {valid_cycles}")
        print(f"  Boundary discards: {boundary_discards}")

    df_cycles = pd.DataFrame(cycle_rows)
    df_summary = pd.DataFrame(summary_rows)

    df_cycles.to_csv(OUT_CSV, index=False)

    save_multi_sheet_excel(
        [("files_summary", df_summary), ("cycles", df_cycles)],
        OUT_XLSX,
    )

    print("\n[OK] Max Spectral Centroid Segmentation finished.")
    print("CSV saved to:", OUT_CSV.resolve())
    print("XLSX saved to:", OUT_XLSX.resolve())
    print(f"Total cycles created: {len(df_cycles)}")


if __name__ == "__main__":
    max_spectral_centroid_segmentation(
        cycle_length_s=2.5,
        peak_offset_before_s=0.8,
        peak_offset_after_s=1.7,
        frame_ms=25.0,
        hop_ms=10.0,
        min_peak_distance_s=1.0,
        min_peak_percentile=75.0
    )
