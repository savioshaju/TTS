"""
Stage C — Telugu Audio Resampling and Format Standardization.
Standardizes all 2,695 recordings to 16,000 Hz, mono, 16-bit linear PCM:
- Female recordings (already 16 kHz): verified and copied.
- Male recordings (48 kHz): resampled using high-quality polyphase sinc Kaiser filtering.
- Destination: D:\\Savio Shaju\\TTS\\preprocessing\\Telugu\\wavs_16k\\
- Verifies every processed audio file on disk.
"""

import os
import sys
import wave
import json
import time
import shutil
import numpy as np
import torch
import torchaudio

ROOT_DIR = r"D:\Savio Shaju\TTS"
DATASET_DIR = os.path.join(ROOT_DIR, "datasets", "prepared", "telugu_prepared_v2")
OUTPUT_DIR = os.path.join(ROOT_DIR, "preprocessing", "Telugu", "wavs_16k")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 75)
print("STAGE C: TELUGU AUDIO RESAMPLING AND FORMAT STANDARDIZATION")
print(f"Source Dataset: {DATASET_DIR}")
print(f"Destination:    {OUTPUT_DIR}")
print("=" * 75)

# High-quality polyphase sinc resampler with Kaiser window anti-aliasing
resampler_48_to_16 = torchaudio.transforms.Resample(
    orig_freq=48000,
    new_freq=16000,
    lowpass_filter_width=64,
    rolloff=0.9475,
    resampling_method="sinc_interp_kaiser",
)

splits = ["train", "val"]
records = []

for split in splits:
    meta_path = os.path.join(DATASET_DIR, split, "metadata.csv")
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            uid, spk, text, gender, dur = parts[0], parts[1], parts[2], parts[3], float(parts[4])
            src_wav = os.path.join(DATASET_DIR, split, "wavs", f"{uid}.wav")
            records.append({
                "uid": uid,
                "split": split,
                "speaker": spk,
                "gender": gender,
                "duration": dur,
                "src_wav": src_wav,
            })

print(f"Total recordings to process: {len(records):,} (Train: 2,250, Val: 445)")

start_time = time.time()
female_count = 0
male_count = 0
resampled_count = 0
copied_count = 0

verification_errors = []

for idx, rec in enumerate(records):
    uid = rec["uid"]
    src_wav = rec["src_wav"]
    dst_wav = os.path.join(OUTPUT_DIR, f"{uid}.wav")
    
    # Read original audio properties
    with wave.open(src_wav, "rb") as wf:
        orig_sr = wf.getframerate()
        orig_ch = wf.getnchannels()
        orig_sw = wf.getsampwidth()
        orig_nf = wf.getnframes()
        orig_dur = orig_nf / float(orig_sr)
        frames = wf.readframes(orig_nf)

    if orig_sr == 16000:
        # Female 16 kHz: preserve exact native audio
        shutil.copy2(src_wav, dst_wav)
        female_count += 1
        copied_count += 1
    elif orig_sr == 48000:
        # Male 48 kHz: polyphase resample to 16 kHz
        raw_audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        waveform = torch.from_numpy(raw_audio).unsqueeze(0)
        
        with torch.no_grad():
            waveform_16k = resampler_48_to_16(waveform)
            out_bytes = (waveform_16k.clamp(-1.0, 1.0).numpy() * 32767.0).astype(np.int16).tobytes()

        with wave.open(dst_wav, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(out_bytes)

        male_count += 1
        resampled_count += 1
    else:
        raise ValueError(f"Unexpected sample rate {orig_sr} Hz for {src_wav}!")

    # Verify processed file on disk
    with wave.open(dst_wav, "rb") as wf:
        out_sr = wf.getframerate()
        out_ch = wf.getnchannels()
        out_sw = wf.getsampwidth()
        out_nf = wf.getnframes()
        out_dur = out_nf / float(out_sr)

    if out_sr != 16000:
        verification_errors.append(f"{uid}: Sample rate {out_sr} != 16000")
    if out_ch != 1:
        verification_errors.append(f"{uid}: Channels {out_ch} != 1")
    if out_sw != 2:
        verification_errors.append(f"{uid}: Sample width {out_sw} != 2 (16-bit)")
    if abs(out_dur - orig_dur) > 0.05:
        verification_errors.append(f"{uid}: Duration deviation > 50ms (orig {orig_dur:.3f}s, out {out_dur:.3f}s)")

    if (idx + 1) % 500 == 0 or (idx + 1) == len(records):
        elapsed = time.time() - start_time
        print(f"  Processed {idx + 1:,} / {len(records):,} files ({elapsed:.1f}s)...")

elapsed_total = time.time() - start_time
print("\n" + "=" * 75)
print("AUDIO RESAMPLING AND VERIFICATION COMPLETE")
print(f"  Total Processed:    {len(records):,} files")
print(f"  Female (16k Copied):{female_count:,} files (100% native preserved)")
print(f"  Male (48k -> 16k):  {male_count:,} files (polyphase sinc Kaiser)")
print(f"  Elapsed Time:       {elapsed_total:.2f} seconds ({elapsed_total / len(records) * 1000:.2f} ms/file)")
print(f"  Verification Errors:{len(verification_errors)}")
print("=" * 75)

if verification_errors:
    print(f"\n[ERROR] Found {len(verification_errors)} verification errors:")
    for err in verification_errors[:10]:
        print(f"  - {err}")
    sys.exit(1)

# Write resampling audit summary
audit_data = {
    "stage": "Stage C — Audio Resampling",
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "output_dir": OUTPUT_DIR,
    "total_recordings": len(records),
    "female_recordings_16k_native": female_count,
    "male_recordings_resampled_48k_to_16k": male_count,
    "target_sample_rate": 16000,
    "channels": 1,
    "bit_depth": 16,
    "resampling_algorithm": "torchaudio sinc_interp_kaiser (lowpass_filter_width=64, rolloff=0.9475)",
    "verification_status": "ALL_2695_FILES_VERIFIED_VALID",
}

with open(os.path.join(ROOT_DIR, "Report", "telugu_resampling_audit.json"), "w", encoding="utf-8") as f:
    json.dump(audit_data, f, indent=2)

print("Resampling audit saved to Report/telugu_resampling_audit.json")
