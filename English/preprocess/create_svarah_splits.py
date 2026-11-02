import os
import io
import wave
import json
import glob
import random
import numpy as np
import polars as pl
from collections import defaultdict

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

BASE_DIR = r"d:\Savio Shaju\TTS"
SVA_DATA_DIR = os.path.join(BASE_DIR, "datasets", "Svarah", "data")
OUT_DIR = os.path.join(BASE_DIR, "datasets", "prepared", "english_speaker_independent")

os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 80)
print("SVARAH SPEAKER-INDEPENDENT DATASET PREPARATION")
print("=" * 80)

sva_files = sorted(glob.glob(os.path.join(SVA_DATA_DIR, "*.parquet")))
print(f"Reading {len(sva_files)} Parquet files from {SVA_DATA_DIR}...")

# Step 1: Collect all utterances with demographic speaker identification
demographic_cols = [
    "gender", "age-group", "primary_language", "native_place_state",
    "native_place_district", "highest_qualification", "job_category", "occupation_domain"
]

all_rows = []
for f_idx, f_path in enumerate(sva_files):
    df = pl.read_parquet(f_path)
    for row_idx, row in enumerate(df.iter_rows(named=True)):
        audio_info = row["audio_filepath"]
        p = audio_info["path"]
        b = audio_info["bytes"]
        
        # Calculate actual audio duration from WAV binary header
        with wave.open(io.BytesIO(b), 'rb') as w:
            n_frames = w.getnframes()
            sr = w.getframerate()
            actual_dur = n_frames / sr
            
        demo_key = tuple(str(row.get(col, "unknown")).strip() for col in demographic_cols)
        
        text = (row.get("text") or "").strip()
        
        all_rows.append({
            "uid": f"svarah_{f_idx}_{row_idx}",
            "filename": p,
            "file_idx": f_idx,
            "row_idx": row_idx,
            "demo_key": demo_key,
            "gender": str(row.get("gender", "unknown")).strip(),
            "age_group": str(row.get("age-group", "unknown")).strip(),
            "language": str(row.get("primary_language", "unknown")).strip(),
            "state": str(row.get("native_place_state", "unknown")).strip(),
            "district": str(row.get("native_place_district", "unknown")).strip(),
            "actual_duration": actual_dur,
            "reported_duration": float(row.get("duration", actual_dur)),
            "text": text,
            "audio_bytes": b
        })

print(f"Total extracted records: {len(all_rows)}")

# Step 2: Assign deterministic Speaker IDs to unique demographic profiles
unique_demo_keys = sorted(list(set(r["demo_key"] for r in all_rows)))
demo_to_spkid = {k: f"spk_{i:03d}" for i, k in enumerate(unique_demo_keys)}

for r in all_rows:
    r["speaker_id"] = demo_to_spkid[r["demo_key"]]

# Group utterances by speaker
speakers = defaultdict(list)
for r in all_rows:
    # Filter TTS sentence durations: 1.2s <= dur <= 12.0s, non-empty text
    if 1.2 <= r["actual_duration"] <= 12.0 and len(r["text"]) >= 3:
        speakers[r["speaker_id"]].append(r)

print(f"Total qualified speakers: {len(speakers)}")

# Calculate total duration per speaker
speaker_summary = []
for spk_id, utts in speakers.items():
    spk_dur = sum(u["actual_duration"] for u in utts)
    meta = utts[0]
    speaker_summary.append({
        "speaker_id": spk_id,
        "gender": meta["gender"],
        "language": meta["language"],
        "state": meta["state"],
        "district": meta["district"],
        "utterance_count": len(utts),
        "total_duration": spk_dur,
        "utterances": utts
    })

# Shuffle with fixed seed
random.seed(SEED)
random.shuffle(speaker_summary)

# Step 3: Stratified Speaker Partitioning
# Target: ~1.0 hour (3600s) for validation, ~5.0 hours (18000s) for training
target_val_sec = 3600.0
target_train_sec = 18000.0

val_speakers = []
train_speakers = []
val_sec = 0.0
train_sec = 0.0

# Stratification key: (gender, primary_language)
strat_buckets = defaultdict(list)
for spk in speaker_summary:
    strat_buckets[(spk["gender"], spk["language"])].append(spk)

# Allocate to Val first to achieve high demographic diversity (~1 hour)
for bucket_key, spk_list in strat_buckets.items():
    for spk in spk_list:
        if val_sec + spk["total_duration"] <= target_val_sec + 120.0:
            val_speakers.append(spk)
            val_sec += spk["total_duration"]
            break

# Allocate remaining speakers to Train to reach ~5 hours
val_spk_ids = set(s["speaker_id"] for s in val_speakers)
for spk in speaker_summary:
    if spk["speaker_id"] not in val_spk_ids:
        if train_sec + spk["total_duration"] <= target_train_sec + 200.0:
            train_speakers.append(spk)
            train_sec += spk["total_duration"]

# If train is slightly under, add next available speakers
for spk in speaker_summary:
    if spk["speaker_id"] not in val_spk_ids and spk not in train_speakers:
        if train_sec < target_train_sec:
            train_speakers.append(spk)
            train_sec += spk["total_duration"]

# Step 4: Verification of Speaker Independence & 0% Overlap
train_spk_set = set(s["speaker_id"] for s in train_speakers)
val_spk_set = set(s["speaker_id"] for s in val_speakers)

assert train_spk_set.isdisjoint(val_spk_set), "CRITICAL: Speaker overlap detected between train and val!"
print("\nSpeaker Independence Verification:")
print(f"  Train Speakers: {len(train_spk_set)}")
print(f"  Val Speakers:   {len(val_spk_set)}")
print(f"  Speaker Intersection: {train_spk_set.intersection(val_spk_set)} (Overlap: 0%)")

# Flatten utterances
train_utts = [u for s in train_speakers for u in s["utterances"]]
val_utts = [u for s in val_speakers for u in s["utterances"]]

train_uids = set(u["uid"] for u in train_utts)
val_uids = set(u["uid"] for u in val_utts)
assert train_uids.isdisjoint(val_uids), "CRITICAL: Utterance overlap detected!"
print(f"  Utterance Intersection: Overlap: 0%")

print(f"\nFinal Split Durations (Calculated from Audio PCM Frames):")
print(f"  Train: {len(train_utts)} samples across {len(train_speakers)} speakers | {train_sec/3600:.2f} hours ({train_sec:.1f} s)")
print(f"  Val:   {len(val_utts)} samples across {len(val_speakers)} speakers   | {val_sec/3600:.2f} hours ({val_sec:.1f} s)")

# Step 5: Export to disk
def export_split(split_name, utterances, split_speakers):
    split_dir = os.path.join(OUT_DIR, split_name)
    wav_dir = os.path.join(split_dir, "wavs")
    os.makedirs(wav_dir, exist_ok=True)
    
    meta_lines = []
    print(f"\nExporting {len(utterances)} audio files for {split_name} split...")
    for u in utterances:
        # Peak normalize audio
        with wave.open(io.BytesIO(u["audio_bytes"]), 'rb') as in_w:
            raw_frames = in_w.readframes(in_w.getnframes())
            audio_data = np.frombuffer(raw_frames, dtype=np.int16).astype(np.float32)
            max_v = np.max(np.abs(audio_data))
            if max_v > 0:
                audio_data = (audio_data / max_v) * 32767.0
            norm_bytes = audio_data.astype(np.int16).tobytes()
            
        out_wav_path = os.path.join(wav_dir, f"{u['uid']}.wav")
        with wave.open(out_wav_path, 'wb') as out_w:
            out_w.setnchannels(1)
            out_w.setsampwidth(2)
            out_w.setframerate(16000)
            out_w.writeframes(norm_bytes)
            
        meta_lines.append(f"{u['uid']}|{u['speaker_id']}|{u['text']}|{u['gender']}|{u['language']}|{u['district']}|{u['actual_duration']:.3f}")
        
    meta_path = os.path.join(split_dir, "metadata.csv")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write("\n".join(meta_lines) + "\n")
        
    # Standard FastSpeech 2 format: basename|speaker|text|raw_text
    txt_path = os.path.join(split_dir, f"{split_name}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for u in utterances:
            f.write(f"{u['uid']}|{u['speaker_id']}|{u['text']}|{u['text']}\n")

export_split("train", train_utts, train_speakers)
export_split("val", val_utts, val_speakers)

# Step 6: Generate Full Summary JSON
def get_dist(speakers):
    genders = defaultdict(float)
    languages = defaultdict(float)
    districts = defaultdict(float)
    for s in speakers:
        genders[s["gender"]] += s["total_duration"]
        languages[s["language"]] += s["total_duration"]
        districts[s["district"]] += s["total_duration"]
    return {
        "speaker_count": len(speakers),
        "genders_hours": {k: round(v / 3600, 2) for k, v in genders.items()},
        "languages_hours": {k: round(v / 3600, 2) for k, v in languages.items()},
        "district_count": len(districts)
    }

summary = {
    "dataset": "ai4bharat/Svarah",
    "official_split_flag": {
        "status": "FLAGGED",
        "details": "Hugging Face ai4bharat/Svarah repository only contains official 'test' split files (data/test-*.parquet, 9.61 total hours). No official 'train' or 'validation' split exists upstream. To satisfy the assignment requirement of ~5h train and ~1h val, a speaker-independent partition was constructed from the available 9.61h pool, strictly preserving the original Parquet files."
    },
    "split_strategy": "Speaker-Independent (Zero Speaker Overlap)",
    "reproducibility_seed": SEED,
    "train_split": {
        "sample_count": len(train_utts),
        "duration_hours": round(train_sec / 3600, 3),
        "duration_seconds": round(train_sec, 1),
        "distribution": get_dist(train_speakers)
    },
    "val_split": {
        "sample_count": len(val_utts),
        "duration_hours": round(val_sec / 3600, 3),
        "duration_seconds": round(val_sec, 1),
        "distribution": get_dist(val_speakers)
    },
    "overlap_verification": {
        "speaker_overlap_count": len(train_spk_set.intersection(val_spk_set)),
        "utterance_overlap_count": len(train_uids.intersection(val_uids)),
        "overlap_percentage": 0.0
    }
}

summary_path = os.path.join(OUT_DIR, "split_summary.json")
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print(f"\nSummary successfully written to: {summary_path}")

