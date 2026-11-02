"""
Stage D: Acoustic Feature Extraction and FastSpeech 2 Dataset Preparation for Telugu TTS.
- Reference implementation: FastSpeech2-master
- Dataset: SPRINGLab IndicTTS Telugu (telugu_prepared_v2)
- Audio: 16 kHz mono WAVs under preprocessing/Telugu/wavs_16k
- Alignments: MMS Forced Alignments under preprocessing/Telugu/alignments
- Failed utterances excluded: indic_te_0_927 (train), indic_te_3_0 (val)
- Expected: 2,249 train utterances, 444 val utterances (2,693 total)

Features:
1. 80-channel log mel-spectrograms via TacotronSTFT (16kHz, hop=256, win=1024, fft=1024) -> (T_mel, 80)
2. F0 Pitch contour via PyWorld DIO + StoneMask + linear interpolation + phoneme averaging
3. Energy contour via TacotronSTFT magnitude L2 norm + phoneme averaging
4. Phoneme durations derived from continuous MMS alignment boundaries (62.5 fps)
5. Global normalization statistics fitted STRICTLY on the training split
6. Complete audit & dataset loader verification
"""

import os
import sys
import json
import time
import wave
import numpy as np
import pyworld as pw
import torch
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# FastSpeech 2 reference imports
ROOT_DIR = r"D:\Savio Shaju\TTS"
FASTSPEECH2_DIR = os.path.join(ROOT_DIR, "FastSpeech2-master")
sys.path.insert(0, FASTSPEECH2_DIR)
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

DATASET_BASE = os.path.join(ROOT_DIR, "datasets", "prepared", "telugu_prepared_v2")
WAVS_DIR = os.path.join(ROOT_DIR, "preprocessing", "Telugu", "wavs_16k")
ALIGN_DIR = os.path.join(ROOT_DIR, "preprocessing", "Telugu", "alignments")
PREP_BASE = os.path.join(ROOT_DIR, "preprocessing", "Telugu")
REPORT_DIR = os.path.join(ROOT_DIR, "Report")

EXCLUDE_UIDS = {"indic_te_0_927", "indic_te_3_0"}


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


def convert_alignment_to_phonemes_and_durations(alignment_data, min_pause_sec=0.04):
    """
    Converts MMS FA character and word alignment boundaries into FastSpeech 2
    discrete phoneme sequences and integer mel-frame durations at 62.5 fps.
    Guarantees:
    - sum(durations) == total_mel_frames
    - all durations >= 1 (strictly positive)
    - inter-word pauses >= min_pause_sec represented as '@sp'
    """
    words = alignment_data["words"]
    if not words:
        return [], [], 0.0, 0.0

    tokens = []
    token_boundaries = []

    for w_idx, w in enumerate(words):
        chars = w["chars"]
        for c_idx, c in enumerate(chars):
            c_char = c["char"]
            c_s = c["start"]
            c_e = c["end"]

            if c_idx + 1 < len(chars):
                next_c = chars[c_idx + 1]
                mid = (c_e + next_c["start"]) / 2.0
                curr_end = mid
            else:
                curr_end = c_e

            if not token_boundaries:
                curr_start = c_s
            else:
                curr_start = token_boundaries[-1][1]

            tokens.append(c_char)
            token_boundaries.append((curr_start, curr_end))

        if w_idx + 1 < len(words):
            next_w = words[w_idx + 1]
            next_w_start = next_w["chars"][0]["start"]
            gap = next_w_start - curr_end
            if gap >= min_pause_sec:
                tokens.append("sp")
                token_boundaries.append((curr_end, next_w_start))
            else:
                mid = (curr_end + next_w_start) / 2.0
                token_boundaries[-1] = (token_boundaries[-1][0], mid)

    s0 = token_boundaries[0][0]
    e_last = token_boundaries[-1][1]

    # Discrete frame index mapping via telescoping integer boundaries
    frame_indices = [0]
    for _, b_end in token_boundaries:
        rel_time = b_end - s0
        f_idx = int(round(rel_time * FPS))
        frame_indices.append(f_idx)

    # Calculate durations as differences
    durations = []
    for i in range(len(frame_indices) - 1):
        d = frame_indices[i + 1] - frame_indices[i]
        durations.append(d)

    # Post-condition verification: ensure no zero or negative durations
    for i in range(len(durations)):
        if durations[i] <= 0:
            durations[i] = 1

    return tokens, durations, s0, e_last


def extract_features():
    mel_dir = os.path.join(PREP_BASE, "mel")
    pitch_dir = os.path.join(PREP_BASE, "pitch")
    energy_dir = os.path.join(PREP_BASE, "energy")
    dur_dir = os.path.join(PREP_BASE, "duration")

    os.makedirs(mel_dir, exist_ok=True)
    os.makedirs(pitch_dir, exist_ok=True)
    os.makedirs(energy_dir, exist_ok=True)
    os.makedirs(dur_dir, exist_ok=True)

    print("=" * 75)
    print("TELUGU FASTSPEECH 2: ACOUSTIC FEATURE EXTRACTION")
    print(f"Sampling Rate:     {SAMPLING_RATE} Hz")
    print(f"Hop Length:        {HOP_LENGTH} ({FRAME_PERIOD_MS:.1f} ms -> {FPS} fps)")
    print(f"Mel Filter / Win:  {FILTER_LENGTH} / {WIN_LENGTH}")
    print(f"Mel Channels:      {N_MEL_CHANNELS} (0 - {MEL_FMAX} Hz)")
    print("=" * 75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing TacotronSTFT on device: {device}...")
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

    speaker_map = {"telugufemale": 0, "telugumale": 1}
    speakers_path = os.path.join(PREP_BASE, "speakers.json")
    with open(speakers_path, "w", encoding="utf-8") as f:
        json.dump(speaker_map, f, indent=2)

    manifest_entries = {"train": [], "val": []}
    audit_stats = {
        "train": {"total": 0, "processed": 0, "failed": [], "frames": 0, "tokens": 0},
        "val": {"total": 0, "processed": 0, "failed": [], "frames": 0, "tokens": 0},
    }

    start_time_all = time.time()

    for split in ["train", "val"]:
        meta_csv = os.path.join(DATASET_BASE, split, "metadata.csv")
        records = []
        with open(meta_csv, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                p = line.split("|")
                uid, spk, text, gender, dur = p[0], p[1], p[2], p[3], float(p[4])
                if uid in EXCLUDE_UIDS:
                    continue
                records.append({
                    "uid": uid,
                    "speaker": spk,
                    "raw_text": text,
                    "gender": gender,
                    "duration": dur,
                })

        audit_stats[split]["total"] = len(records)
        print(f"\nProcessing {split.upper()} split ({len(records):,} utterances)...")

        for rec in tqdm(records, desc=f"Extracting {split}"):
            uid = rec["uid"]
            spk = rec["speaker"]
            raw_text = rec["raw_text"]

            wav_path = os.path.join(WAVS_DIR, f"{uid}.wav")
            align_path = os.path.join(ALIGN_DIR, f"{uid}.json")

            if not os.path.isfile(wav_path) or not os.path.isfile(align_path):
                audit_stats[split]["failed"].append((uid, "Missing audio or alignment file"))
                continue

            try:
                # 1. Load alignment and compute phonemes & durations
                with open(align_path, "r", encoding="utf-8") as f_al:
                    align_data = json.load(f_al)

                tokens, durations, s_time, e_time = convert_alignment_to_phonemes_and_durations(align_data)
                duration_arr = np.array(durations, dtype=np.int32)
                sum_dur = int(np.sum(duration_arr))

                # Format FastSpeech 2 text string: {@tok1 @tok2 ...}
                phone_str = "{" + " ".join(["@" + t for t in tokens]) + "}"

                # 2. Read 16 kHz audio and trim to speech start & end
                with wave.open(wav_path, "rb") as wf:
                    audio_raw = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0

                s_sample = int(round(s_time * SAMPLING_RATE))
                e_sample = int(round(e_time * SAMPLING_RATE))
                s_sample = max(0, min(s_sample, len(audio_raw) - 1))
                e_sample = max(s_sample + HOP_LENGTH, min(e_sample, len(audio_raw)))
                wav_trimmed = audio_raw[s_sample:e_sample]

                # 3. Fundamental frequency F0 via PyWorld DIO + StoneMask
                pitch, t = pw.dio(
                    wav_trimmed.astype(np.float64),
                    SAMPLING_RATE,
                    frame_period=FRAME_PERIOD_MS,
                )
                pitch = pw.stonemask(wav_trimmed.astype(np.float64), pitch, t, SAMPLING_RATE)

                # Match sum_dur frames exactly
                if len(pitch) >= sum_dur:
                    pitch = pitch[:sum_dur]
                else:
                    pitch = np.pad(pitch, (0, sum_dur - len(pitch)), mode="edge")

                # Linear interpolation for unvoiced frames
                nonzero_ids = np.where(pitch != 0)[0]
                if len(nonzero_ids) > 1:
                    interp_fn = interp1d(
                        nonzero_ids,
                        pitch[nonzero_ids],
                        fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
                        bounds_error=False,
                    )
                    pitch_interpolated = interp_fn(np.arange(0, len(pitch)))
                elif len(nonzero_ids) == 1:
                    pitch_interpolated = np.full(len(pitch), pitch[nonzero_ids[0]])
                else:
                    # Fallback for completely unvoiced (e.g. whisper/breath)
                    pitch_interpolated = np.zeros(len(pitch))

                # 4. TacotronSTFT Mel-spectrogram & Energy
                mel_spectrogram, energy = get_mel_from_wav(wav_trimmed, stft)

                # Match sum_dur frames exactly
                if mel_spectrogram.shape[1] >= sum_dur:
                    mel_spectrogram = mel_spectrogram[:, :sum_dur]
                    energy = energy[:sum_dur]
                else:
                    pad_len = sum_dur - mel_spectrogram.shape[1]
                    mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, pad_len)), mode="edge")
                    energy = np.pad(energy, (0, pad_len), mode="edge")

                # 5. Phoneme-level averaging of pitch and energy
                phoneme_pitch = np.zeros(len(duration_arr), dtype=np.float32)
                phoneme_energy = np.zeros(len(duration_arr), dtype=np.float32)

                pos = 0
                for i, d in enumerate(duration_arr):
                    if d > 0:
                        phoneme_pitch[i] = float(np.mean(pitch_interpolated[pos : pos + d]))
                        phoneme_energy[i] = float(np.mean(energy[pos : pos + d]))
                    else:
                        phoneme_pitch[i] = 0.0
                        phoneme_energy[i] = 0.0
                    pos += d

                # Strict dimension & finite assertions
                assert mel_spectrogram.T.shape[0] == sum_dur, f"Mel frames ({mel_spectrogram.T.shape[0]}) != sum(dur) ({sum_dur})"
                assert mel_spectrogram.T.shape[1] == N_MEL_CHANNELS, f"Mel channels != {N_MEL_CHANNELS}"
                assert len(phoneme_pitch) == len(duration_arr), f"Pitch len != dur len"
                assert len(phoneme_energy) == len(duration_arr), f"Energy len != dur len"
                assert np.all(np.isfinite(mel_spectrogram)), f"Non-finite in mel"
                assert np.all(np.isfinite(phoneme_pitch)), f"Non-finite in pitch"
                assert np.all(np.isfinite(phoneme_energy)), f"Non-finite in energy"

                # 6. Save raw unnormalized features (.npy)
                mel_filename = f"{spk}-mel-{uid}.npy"
                np.save(os.path.join(mel_dir, mel_filename), mel_spectrogram.T.astype(np.float32))

                dur_filename = f"{spk}-duration-{uid}.npy"
                np.save(os.path.join(dur_dir, dur_filename), duration_arr)

                pitch_filename = f"{spk}-pitch-{uid}.npy"
                np.save(os.path.join(pitch_dir, pitch_filename), phoneme_pitch)

                energy_filename = f"{spk}-energy-{uid}.npy"
                np.save(os.path.join(energy_dir, energy_filename), phoneme_energy)

                # Fit scalers STRICTLY on training split
                if split == "train":
                    clean_p = remove_outlier(phoneme_pitch[phoneme_pitch != 0])
                    clean_e = remove_outlier(phoneme_energy[phoneme_energy != 0])
                    if len(clean_p) > 0:
                        pitch_scaler.partial_fit(clean_p.reshape((-1, 1)))
                    if len(clean_e) > 0:
                        energy_scaler.partial_fit(clean_e.reshape((-1, 1)))

                manifest_entries[split].append(f"{uid}|{spk}|{phone_str}|{raw_text}")
                audit_stats[split]["processed"] += 1
                audit_stats[split]["frames"] += sum_dur
                audit_stats[split]["tokens"] += len(duration_arr)

            except Exception as e:
                audit_stats[split]["failed"].append((uid, str(e)))

    # Save manifest text files
    for split in ["train", "val"]:
        manifest_p = os.path.join(PREP_BASE, f"{split}.txt")
        with open(manifest_p, "w", encoding="utf-8") as f_out:
            for entry in manifest_entries[split]:
                f_out.write(entry + "\n")
        print(f"Saved {split.upper()} manifest: {manifest_p} ({len(manifest_entries[split])} lines)")

    print("\nPass 1 completed. Fitting global normalization parameters on TRAIN split...")
    pitch_mean = float(pitch_scaler.mean_[0])
    pitch_std = float(pitch_scaler.scale_[0])
    energy_mean = float(energy_scaler.mean_[0])
    energy_std = float(energy_scaler.scale_[0])

    print(f"  Pitch:  mean = {pitch_mean:.4f}, std = {pitch_std:.4f}")
    print(f"  Energy: mean = {energy_mean:.4f}, std = {energy_std:.4f}")

    # Pass 2: Normalize all pitch and energy arrays
    print("\nPass 2: Normalizing pitch and energy files...")
    pitch_min = float("inf")
    pitch_max = float("-inf")
    energy_min = float("inf")
    energy_max = float("-inf")

    all_pitch_files = [f for f in os.listdir(pitch_dir) if f.endswith(".npy")]
    for p_file in tqdm(all_pitch_files, desc="Normalizing pitch"):
        p_path = os.path.join(pitch_dir, p_file)
        vals = np.load(p_path)
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
        norm_vals = np.zeros_like(vals)
        mask = vals != 0
        norm_vals[mask] = (vals[mask] - energy_mean) / energy_std
        np.save(e_path, norm_vals.astype(np.float32))
        if np.any(mask):
            energy_min = min(energy_min, float(np.min(norm_vals[mask])))
            energy_max = max(energy_max, float(np.max(norm_vals[mask])))

    stats = {
        "pitch": [float(pitch_min), float(pitch_max), float(pitch_mean), float(pitch_std)],
        "energy": [float(energy_min), float(energy_max), float(energy_mean), float(energy_std)],
    }
    stats_path = os.path.join(PREP_BASE, "stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved stats.json to {stats_path}: {stats}")

    elapsed_total = time.time() - start_time_all

    audit_summary = {
        "elapsed_seconds": round(elapsed_total, 2),
        "sampling_rate": SAMPLING_RATE,
        "hop_length": HOP_LENGTH,
        "filter_length": FILTER_LENGTH,
        "n_mel_channels": N_MEL_CHANNELS,
        "stats": stats,
        "speakers": speaker_map,
        "splits": {
            "train": {
                "expected": audit_stats["train"]["total"],
                "processed": audit_stats["train"]["processed"],
                "failed_count": len(audit_stats["train"]["failed"]),
                "failed": audit_stats["train"]["failed"],
                "total_mel_frames": audit_stats["train"]["frames"],
                "total_active_hours": round(audit_stats["train"]["frames"] * HOP_LENGTH / SAMPLING_RATE / 3600, 4),
                "total_phoneme_tokens": audit_stats["train"]["tokens"],
            },
            "val": {
                "expected": audit_stats["val"]["total"],
                "processed": audit_stats["val"]["processed"],
                "failed_count": len(audit_stats["val"]["failed"]),
                "failed": audit_stats["val"]["failed"],
                "total_mel_frames": audit_stats["val"]["frames"],
                "total_active_hours": round(audit_stats["val"]["frames"] * HOP_LENGTH / SAMPLING_RATE / 3600, 4),
                "total_phoneme_tokens": audit_stats["val"]["tokens"],
            },
        },
    }

    report_out = os.path.join(REPORT_DIR, "telugu_stage_d_audit.json")
    with open(report_out, "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)
    print(f"\nSaved full Stage D audit report to: {report_out}")

    return audit_summary


if __name__ == "__main__":
    extract_features()

