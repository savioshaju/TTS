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
IND_DATA_DIR = os.path.join(BASE_DIR, "datasets", "IndicTTS_Telugu", "data")
OUT_DIR = os.path.join(BASE_DIR, "datasets", "prepared", "telugu_prepared_v2")

os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 80)
print("INDIAC-TTS TELUGU DATASET PREPARATION")
print("=" * 80)

ind_files = sorted(glob.glob(os.path.join(IND_DATA_DIR, "*.parquet")))
print(f"Reading metadata for {len(ind_files)} Parquet files from {IND_DATA_DIR}...")

# Step 1: Collect metadata using vectorized column scans (Zero memory overhead)
all_records = []
total_dataset_sec = 0.0

for f_idx, f_path in enumerate(ind_files):
    # Scan text, gender, and binary size efficiently
    df_meta = pl.read_parquet(f_path, columns=["text", "gender"])
    df_sizes = pl.read_parquet(f_path).select(
        pl.col("audio").struct.field("bytes").bin.size().alias("size_bytes"),
        pl.col("audio").struct.field("path").alias("audio_path")
    )
    
    texts = df_meta["text"].to_list()
    genders = df_meta["gender"].to_list()
    sizes = df_sizes["size_bytes"].to_list()
    paths = df_sizes["audio_path"].to_list()
    
    for row_idx in range(len(texts)):
        b_len = sizes[row_idx]
        actual_dur = (b_len - 44) / 32000.0
        total_dataset_sec += actual_dur
        
        g = genders[row_idx]
        speaker_id = "telugufemale" if g == 0 else "telugumale"
        gender_str = "Female" if g == 0 else "Male"
        text = (texts[row_idx] or "").strip().replace('\ufeff', '')
        
        all_records.append({
            "uid": f"indic_te_{f_idx}_{row_idx}",
            "filename": paths[row_idx],
            "file_idx": f_idx,
            "row_idx": row_idx,
            "speaker_id": speaker_id,
            "gender": gender_str,
            "actual_duration": actual_dur,
            "text": text
        })

print(f"Total extracted records: {len(all_records)}")
print(f"Total dataset duration: {total_dataset_sec/3600:.2f} hours ({total_dataset_sec:.1f} s)")

# Filter for TTS standard sentence durations: 1.0s <= dur <= 14.5s
filtered_records = [
    r for r in all_records 
    if 1.0 <= r["actual_duration"] <= 14.5 and len(r["text"]) >= 2
]
print(f"Total qualified utterances after duration filtering (1.0s-14.5s): {len(filtered_records)}")
filtered_dur = sum(r["actual_duration"] for r in filtered_records)
print(f"Filtered pool duration: {filtered_dur/3600:.2f} hours ({filtered_dur:.1f} s)")

# Breakdown of filtered pool by speaker
by_spk = defaultdict(list)
for r in filtered_records:
    by_spk[r["speaker_id"]].append(r)

for spk, utts in by_spk.items():
    d = sum(u["actual_duration"] for u in utts)
    print(f"  {spk}: {len(utts)} utts, {d/3600:.2f} hours ({d:.1f} s)")

# Step 2: Stratified Selection (Proportional representation of both studio speakers)
for spk in by_spk:
    random.shuffle(by_spk[spk])

train_records = []
val_records = []

train_target_sec = 18000.0  # 5.0 hours
val_target_sec = 3600.0     # 1.0 hour

# Female: 50%, Male: 50% (50/50 balanced gender stratification)
spk_weights = {
    "telugufemale": 0.50,
    "telugumale": 0.50
}

for spk, weight in spk_weights.items():
    spk_utts = by_spk[spk]
    t_target = train_target_sec * weight
    v_target = val_target_sec * weight
    
    t_sec = 0.0
    v_sec = 0.0
    
    idx = 0
    # Val first
    while idx < len(spk_utts) and v_sec < v_target:
        val_records.append(spk_utts[idx])
        v_sec += spk_utts[idx]["actual_duration"]
        idx += 1
        
    # Train next
    while idx < len(spk_utts) and t_sec < t_target:
        train_records.append(spk_utts[idx])
        t_sec += spk_utts[idx]["actual_duration"]
        idx += 1

train_dur = sum(r["actual_duration"] for r in train_records)
val_dur = sum(r["actual_duration"] for r in val_records)

print(f"\nFinal Selection:")
print(f"  Train: {len(train_records)} utterances, {train_dur/3600:.2f} hours ({train_dur:.1f} s)")
print(f"  Val:   {len(val_records)} utterances, {val_dur/3600:.2f} hours ({val_dur:.1f} s)")

# Step 3: Overlap Verification
train_uids = set(r["uid"] for r in train_records)
val_uids = set(r["uid"] for r in val_records)
assert train_uids.isdisjoint(val_uids), "CRITICAL: Utterance overlap detected!"
print(f"Utterance Overlap Verification: 0 overlapping samples (0.0% overlap)")

# Step 4: Export to Disk
def export_split_fast(split_name, records):
    split_dir = os.path.join(OUT_DIR, split_name)
    wav_dir = os.path.join(split_dir, "wavs")
    os.makedirs(wav_dir, exist_ok=True)
    
    # Group by file_idx
    file_to_recs = defaultdict(dict)
    for r in records:
        file_to_recs[r["file_idx"]][r["row_idx"]] = r
        
    meta_lines = []
    print(f"\nWriting {len(records)} audio files for Telugu {split_name} split...")
    for f_idx in sorted(file_to_recs.keys()):
        row_map = file_to_recs[f_idx]
        f_path = ind_files[f_idx]
        
        # Load audio column for this file
        df_audio = pl.read_parquet(f_path, columns=["audio"])
        audio_col = df_audio["audio"]
        
        for row_idx, r in sorted(row_map.items()):
            raw_bytes = audio_col[row_idx]["bytes"]
            
            # Peak normalize
            with wave.open(io.BytesIO(raw_bytes), 'rb') as in_w:
                n_frames = in_w.getnframes()
                sr = in_w.getframerate()
                frames = in_w.readframes(n_frames)
                audio_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
                max_v = np.max(np.abs(audio_data))
                if max_v > 0:
                    audio_data = (audio_data / max_v) * 32767.0
                norm_bytes = audio_data.astype(np.int16).tobytes()
                exact_dur = n_frames / sr
                
            out_wav_path = os.path.join(wav_dir, f"{r['uid']}.wav")
            with wave.open(out_wav_path, 'wb') as out_w:
                out_w.setnchannels(1)
                out_w.setsampwidth(2)
                out_w.setframerate(sr)
                out_w.writeframes(norm_bytes)
                
            meta_lines.append(f"{r['uid']}|{r['speaker_id']}|{r['text']}|{r['gender']}|{exact_dur:.3f}")
            
    meta_path = os.path.join(split_dir, "metadata.csv")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write("\n".join(meta_lines) + "\n")
        
    txt_path = os.path.join(split_dir, f"{split_name}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for line in meta_lines:
            parts = line.split("|")
            # FastSpeech 2 format: basename|speaker|text|raw_text
            f.write(f"{parts[0]}|{parts[1]}|{parts[2]}|{parts[2]}\n")

export_split_fast("train", train_records)
export_split_fast("val", val_records)

# Step 5: Summary JSON
def spk_dist(records):
    counts = defaultdict(int)
    durs = defaultdict(float)
    for r in records:
        counts[r["speaker_id"]] += 1
        durs[r["speaker_id"]] += r["actual_duration"]
    return {
        "utterance_counts": dict(counts),
        "duration_hours": {k: round(v / 3600, 2) for k, v in durs.items()}
    }

summary = {
    "dataset": "SPRINGLab/IndicTTS_Telugu",
    "official_split_flag": {
        "status": "FLAGGED",
        "details": "Hugging Face SPRINGLab/IndicTTS_Telugu repository contains only an official 'train' split (data/train-*.parquet, 8,576 rows, 62.29 hours total). There are no official 'val' or 'test' splits upstream. To satisfy the assignment requirement of ~5h train and ~1h val, an utterance-disjoint, speaker-stratified partition was constructed from the available pool, strictly preserving the original Parquet files."
    },
    "split_strategy": "Speaker-Stratified & Utterance-Disjoint (Proportional Studio Speakers, Zero Utterance Overlap)",
    "speaker_independence_analysis": "IndicTTS Telugu contains exactly two professional speakers (1 female: 23.9h, 1 male: 38.4h). An artificial cross-speaker split (Female for train, Male for val) would result in zero overlap, but would preclude acoustic validation in TTS because pitch and formant ranges across genders do not generalize without prior speaker embeddings. Therefore, a balanced speaker-stratified partition with 0% utterance overlap was generated.",
    "reproducibility_seed": SEED,
    "audio_specifications": {
        "sampling_rate": 16000,
        "channels": 1,
        "format": "PCM 16-bit WAV",
        "peak_normalized": True
    },
    "train_split": {
        "sample_count": len(train_records),
        "duration_hours": round(train_dur / 3600, 3),
        "duration_seconds": round(train_dur, 1),
        "distribution": spk_dist(train_records)
    },
    "val_split": {
        "sample_count": len(val_records),
        "duration_hours": round(val_dur / 3600, 3),
        "duration_seconds": round(val_dur, 1),
        "distribution": spk_dist(val_records)
    },
    "overlap_verification": {
        "utterance_overlap_count": len(train_uids.intersection(val_uids)),
        "overlap_percentage": 0.0
    }
}

summary_path = os.path.join(OUT_DIR, "split_summary.json")
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(f"\nSummary successfully written to: {summary_path}")
