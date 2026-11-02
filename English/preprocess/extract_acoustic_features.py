"""
Stage C: Acoustic Feature Extraction for Svarah English TTS.
Extracts:
1. Log-mel spectrograms (80 channels, 16 kHz, filter=1024, win=1024, hop=256)
2. Fundamental frequency (F0) using PyWorld DIO + StoneMask + Linear Interpolation
3. Frame-level energy using L2 norm of STFT magnitude
4. Phoneme-level pitch and energy averaging aligned with durations
5. Global normalization statistics computed strictly from the training split
6. Verification: sum(duration) == T_mel, len(pitch) == len(duration), no NaNs/Infs
"""
import os
import sys
import json
import time
import numpy as np
import soundfile as sf
import pyworld as pw
import torch
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from praatio import textgrid

# FastSpeech 2 imports
ROOT_DIR = r"D:\Savio Shaju\TTS"
sys.path.append(os.path.join(ROOT_DIR, "FastSpeech2-master"))
from audio.stft import TacotronSTFT
from audio.tools import get_mel_from_wav

SAMPLING_RATE = 16000
HOP_LENGTH = 256
FILTER_LENGTH = 1024
WIN_LENGTH = 1024
N_MEL_CHANNELS = 80
MEL_FMIN = 0
MEL_FMAX = 8000
FRAME_PERIOD_MS = (HOP_LENGTH / SAMPLING_RATE) * 1000.0  # 16.0 ms
FPS = SAMPLING_RATE / HOP_LENGTH  # 62.5

DATASET_BASE = os.path.join(ROOT_DIR, "datasets", "prepared", "english_speaker_independent")
PREP_BASE = os.path.join(ROOT_DIR, "preprocessing", "English")

def remove_outlier(values):
    values = np.array(values)
    if len(values) == 0:
        return values
    p25 = np.percentile(values, 25)
    p75 = np.percentile(values, 75)
    lower = p25 - 1.5 * (p75 - p25)
    upper = p75 + 1.5 * (p75 - p25)
    normal_indices = np.logical_and(values > lower, values < upper)
    return values[normal_indices]

def extract_features():
    mel_dir = os.path.join(PREP_BASE, "mel")
    pitch_dir = os.path.join(PREP_BASE, "pitch")
    energy_dir = os.path.join(PREP_BASE, "energy")
    dur_dir = os.path.join(PREP_BASE, "duration")

    os.makedirs(mel_dir, exist_ok=True)
    os.makedirs(pitch_dir, exist_ok=True)
    os.makedirs(energy_dir, exist_ok=True)

    # Initialize STFT
    print("Initializing TacotronSTFT module...")
    stft = TacotronSTFT(
        filter_length=FILTER_LENGTH,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        n_mel_channels=N_MEL_CHANNELS,
        sampling_rate=SAMPLING_RATE,
        mel_fmin=MEL_FMIN,
        mel_fmax=MEL_FMAX,
    )

    pitch_scaler = StandardScaler()
    energy_scaler = StandardScaler()

    splits = ["train", "val"]
    results = {
        "train": {"total": 0, "success": 0, "failed": [], "frames": 0},
        "val": {"total": 0, "success": 0, "failed": [], "frames": 0}
    }

    start_time_all = time.time()
    sil_phones = {"sil", "sp", "spn", ""}

    # 1. First pass: extract and save raw (unnormalized) features, fit scaler on TRAIN only
    for split in splits:
        txt_path = os.path.join(PREP_BASE, f"{split}.txt")
        wav_dir = os.path.join(DATASET_BASE, split, "wavs")
        tg_dir = os.path.join(PREP_BASE, "TextGrid", split)

        print(f"\n{'='*60}")
        print(f"Extracting features for {split.upper()} split...")
        print(f"{'='*60}")

        entries = []
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("|")
                if len(parts) >= 4:
                    entries.append({
                        "base": parts[0],
                        "speaker": parts[1],
                        "phones": parts[2],
                        "raw_text": parts[3]
                    })

        results[split]["total"] = len(entries)

        for entry in tqdm(entries, desc=f"Extracting {split}"):
            base = entry["base"]
            spk = entry["speaker"]

            wav_path = os.path.join(wav_dir, f"{base}.wav")
            tg_path = os.path.join(tg_dir, spk, f"{base}.TextGrid")
            dur_path = os.path.join(dur_dir, f"{spk}-duration-{base}.npy")

            if not os.path.exists(wav_path) or not os.path.exists(tg_path) or not os.path.exists(dur_path):
                results[split]["failed"].append((base, "Missing input file"))
                continue

            try:
                # Load durations
                duration = np.load(dur_path)
                sum_dur = int(np.sum(duration))

                # Load TextGrid to find speech start and end
                tg = textgrid.openTextgrid(tg_path, includeEmptyIntervals=True)
                phone_tier = tg.getTier("phones")

                speech_start = 0.0
                speech_end = tg.maxTimestamp
                started = False

                for p_entry in phone_tier.entries:
                    p = p_entry.label.strip()
                    if not started:
                        if p in sil_phones or not p:
                            continue
                        speech_start = p_entry.start
                        started = True
                    if p not in sil_phones and p:
                        speech_end = p_entry.end

                # Load and slice audio
                wav_full, sr = sf.read(wav_path)
                s_idx = int(np.round(speech_start * SAMPLING_RATE))
                e_idx = int(np.round(speech_end * SAMPLING_RATE))
                wav_trimmed = wav_full[s_idx:e_idx].astype(np.float32)

                if len(wav_trimmed) < HOP_LENGTH:
                    results[split]["failed"].append((base, "Audio trimmed too short"))
                    continue

                # Fundamental frequency F0 via PyWorld DIO + StoneMask
                pitch, t = pw.dio(
                    wav_trimmed.astype(np.float64),
                    SAMPLING_RATE,
                    frame_period=FRAME_PERIOD_MS,
                )
                pitch = pw.stonemask(wav_trimmed.astype(np.float64), pitch, t, SAMPLING_RATE)

                # Truncate or pad pitch to match sum_dur exactly
                if len(pitch) >= sum_dur:
                    pitch = pitch[:sum_dur]
                else:
                    pitch = np.pad(pitch, (0, sum_dur - len(pitch)), mode="edge")

                # Interpolate unvoiced frames
                nonzero_ids = np.where(pitch != 0)[0]
                if len(nonzero_ids) <= 1:
                    # Pitch extraction failed to find voiced frames
                    results[split]["failed"].append((base, "Insufficient voiced pitch frames"))
                    continue

                interp_fn = interp1d(
                    nonzero_ids,
                    pitch[nonzero_ids],
                    fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
                    bounds_error=False,
                )
                pitch_interpolated = interp_fn(np.arange(0, len(pitch)))

                # STFT Mel-spectrogram & Energy
                mel_spectrogram, energy = get_mel_from_wav(wav_trimmed, stft)

                # Match sum_dur frames exactly
                if mel_spectrogram.shape[1] >= sum_dur:
                    mel_spectrogram = mel_spectrogram[:, :sum_dur]
                    energy = energy[:sum_dur]
                else:
                    pad_len = sum_dur - mel_spectrogram.shape[1]
                    mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, pad_len)), mode="edge")
                    energy = np.pad(energy, (0, pad_len), mode="edge")

                # Phoneme-level averaging of pitch and energy
                phoneme_pitch = np.zeros(len(duration), dtype=np.float32)
                phoneme_energy = np.zeros(len(duration), dtype=np.float32)

                pos = 0
                for i, d in enumerate(duration):
                    if d > 0:
                        phoneme_pitch[i] = float(np.mean(pitch_interpolated[pos : pos + d]))
                        phoneme_energy[i] = float(np.mean(energy[pos : pos + d]))
                    else:
                        phoneme_pitch[i] = 0.0
                        phoneme_energy[i] = 0.0
                    pos += d

                # Strict sanity checks
                assert mel_spectrogram.T.shape[0] == sum_dur, f"Mel frames ({mel_spectrogram.T.shape[0]}) != sum(dur) ({sum_dur})"
                assert len(phoneme_pitch) == len(duration), f"Pitch len ({len(phoneme_pitch)}) != dur len ({len(duration)})"
                assert len(phoneme_energy) == len(duration), f"Energy len ({len(phoneme_energy)}) != dur len ({len(duration)})"
                assert not np.isnan(mel_spectrogram).any(), "NaN in mel spectrogram"
                assert not np.isnan(phoneme_pitch).any(), "NaN in phoneme pitch"
                assert not np.isnan(phoneme_energy).any(), "NaN in phoneme energy"

                # Save raw mel-spectrogram: shape (T_mel, 80)
                mel_filename = f"{spk}-mel-{base}.npy"
                np.save(os.path.join(mel_dir, mel_filename), mel_spectrogram.T.astype(np.float32))

                # Save unnormalized phoneme pitch & energy
                pitch_filename = f"{spk}-pitch-{base}.npy"
                np.save(os.path.join(pitch_dir, pitch_filename), phoneme_pitch)

                energy_filename = f"{spk}-energy-{base}.npy"
                np.save(os.path.join(energy_dir, energy_filename), phoneme_energy)

                # Fit scaler ONLY on training split
                if split == "train":
                    clean_p = remove_outlier(phoneme_pitch[phoneme_pitch != 0])
                    clean_e = remove_outlier(phoneme_energy[phoneme_energy != 0])
                    if len(clean_p) > 0:
                        pitch_scaler.partial_fit(clean_p.reshape((-1, 1)))
                    if len(clean_e) > 0:
                        energy_scaler.partial_fit(clean_e.reshape((-1, 1)))

                results[split]["success"] += 1
                results[split]["frames"] += sum_dur

            except Exception as e:
                results[split]["failed"].append((base, str(e)))

    print("\nFeature extraction pass 1 completed.")
    print(f"Train success: {results['train']['success']}/{results['train']['total']}")
    print(f"Val success: {results['val']['success']}/{results['val']['total']}")

    # 2. Compute global normalization stats from the scaler (FITTED ON TRAIN ONLY)
    pitch_mean = float(pitch_scaler.mean_[0])
    pitch_std = float(pitch_scaler.scale_[0])
    energy_mean = float(energy_scaler.mean_[0])
    energy_std = float(energy_scaler.scale_[0])

    print(f"\nGlobal Normalization Parameters (fitted strictly on training split):")
    print(f"  Pitch:  mean = {pitch_mean:.4f}, std = {pitch_std:.4f}")
    print(f"  Energy: mean = {energy_mean:.4f}, std = {energy_std:.4f}")

    # 3. Second pass: normalize all pitch and energy files
    print("\nNormalizing pitch and energy files...")
    pitch_min = float("inf")
    pitch_max = float("-inf")
    energy_min = float("inf")
    energy_max = float("-inf")

    all_pitch_files = [f for f in os.listdir(pitch_dir) if f.endswith(".npy")]
    for p_file in tqdm(all_pitch_files, desc="Normalizing pitch"):
        p_path = os.path.join(pitch_dir, p_file)
        vals = np.load(p_path)
        # Normalize non-zero pitch values
        norm_vals = np.zeros_like(vals)
        mask = vals != 0
        norm_vals[mask] = (vals[mask] - pitch_mean) / pitch_std
        np.save(p_path, norm_vals.astype(np.float32))

        if np.any(mask):
            pitch_min = min(pitch_min, float(np.min(norm_vals[mask])))
            pitch_max = max(pitch_max, float(np.max(norm_vals[mask])))

    all_energy_files = [f for f in os.listdir(energy_dir) if f.endswith(".npy")]
    for e_file in tqdm(all_energy_files, desc="Normalizing energy"):
        e_path = os.path.join(energy_dir, e_file)
        vals = np.load(e_path)
        # Normalize non-zero energy values
        norm_vals = np.zeros_like(vals)
        mask = vals != 0
        norm_vals[mask] = (vals[mask] - energy_mean) / energy_std
        np.save(e_path, norm_vals.astype(np.float32))

        if np.any(mask):
            energy_min = min(energy_min, float(np.min(norm_vals[mask])))
            energy_max = max(energy_max, float(np.max(norm_vals[mask])))

    stats = {
        "pitch": [pitch_min, pitch_max, pitch_mean, pitch_std],
        "energy": [energy_min, energy_max, energy_mean, energy_std]
    }

    stats_path = os.path.join(PREP_BASE, "stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved normalization statistics to: {stats_path}")

    # Save summary report
    elapsed_total = time.time() - start_time_all
    report = {
        "elapsed_seconds": round(elapsed_total, 2),
        "sampling_rate": SAMPLING_RATE,
        "hop_length": HOP_LENGTH,
        "filter_length": FILTER_LENGTH,
        "n_mel_channels": N_MEL_CHANNELS,
        "stats": stats,
        "results": {
            "train": {
                "total": results["train"]["total"],
                "success": results["train"]["success"],
                "failed_count": len(results["train"]["failed"]),
                "failed": results["train"]["failed"],
                "total_mel_frames": results["train"]["frames"],
                "total_active_hours": round(results["train"]["frames"] * HOP_LENGTH / SAMPLING_RATE / 3600, 3)
            },
            "val": {
                "total": results["val"]["total"],
                "success": results["val"]["success"],
                "failed_count": len(results["val"]["failed"]),
                "failed": results["val"]["failed"],
                "total_mel_frames": results["val"]["frames"],
                "total_active_hours": round(results["val"]["frames"] * HOP_LENGTH / SAMPLING_RATE / 3600, 3)
            }
        }
    }

    report_path = os.path.join(PREP_BASE, "reports", "feature_extraction_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved feature extraction report to: {report_path}")

    return report

if __name__ == "__main__":
    extract_features()

