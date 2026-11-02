import os
import io
import wave
import glob
import re
import polars as pl
from collections import defaultdict

BASE_DIR = r"d:\Savio Shaju\TTS"
SVA_DATA_DIR = os.path.join(BASE_DIR, "datasets", "Svarah", "data")
sva_files = sorted(glob.glob(os.path.join(SVA_DATA_DIR, "*.parquet")))

print(f"Reading {len(sva_files)} Svarah parquet files...")

speaker_records = []
total_actual_sec = 0.0
total_records = 0

for f_idx, f in enumerate(sva_files):
    df = pl.read_parquet(f)
    total_records += len(df)
    for row_idx, row in enumerate(df.iter_rows(named=True)):
        audio_info = row["audio_filepath"]
        p = audio_info["path"]
        b = audio_info["bytes"]
        
        # Calculate actual audio duration from wave header
        with wave.open(io.BytesIO(b), 'rb') as w:
            n_frames = w.getnframes()
            sr = w.getframerate()
            actual_dur = n_frames / sr
            
        total_actual_sec += actual_dur
        
        # Extract speaker id: e.g. 281474976884635_f3269_chunk_0.wav -> speaker is f3269 or session prefix
        m = re.search(r'_([fm]\d+)_chunk', p)
        if m:
            speaker_id = m.group(1)
        else:
            parts = p.split("_")
            speaker_id = parts[1] if len(parts) >= 2 else "unknown"
            
        session_id = p.split("_")[0]
        
        speaker_records.append({
            "id": f"svarah_{f_idx}_{row_idx}",
            "filename": p,
            "file_idx": f_idx,
            "row_idx": row_idx,
            "speaker_id": speaker_id,
            "session_id": session_id,
            "gender": row["gender"],
            "age_group": row.get("age-group", "unknown"),
            "language": row.get("primary_language", "unknown"),
            "state": row.get("native_place_state", "unknown"),
            "district": row.get("native_place_district", "unknown"),
            "duration": actual_dur,
            "text": row["text"].strip() if row["text"] else ""
        })

df_all = pl.DataFrame(speaker_records)
print(f"Total rows: {total_records}")
print(f"Total actual duration: {total_actual_sec / 3600:.2f} hours ({total_actual_sec:.1f} s)")

# Group by speaker_id
speaker_stats = df_all.group_by("speaker_id").agg([
    pl.len().alias("utterances"),
    pl.col("duration").sum().alias("total_sec"),
    pl.col("gender").first().alias("gender"),
    pl.col("language").first().alias("language"),
    pl.col("state").first().alias("state"),
    pl.col("district").first().alias("district"),
])

print(f"Unique speaker IDs (by f/m code): {len(speaker_stats)}")

# Also let's check session_id
session_stats = df_all.group_by("session_id").agg([
    pl.len().alias("utterances"),
    pl.col("duration").sum().alias("total_sec"),
    pl.col("gender").first().alias("gender"),
    pl.col("language").first().alias("language"),
])
print(f"Unique session IDs: {len(session_stats)}")

print("\nTop 15 speakers by duration:")
for row in speaker_stats.sort("total_sec", descending=True).head(15).iter_rows(named=True):
    print(f"Speaker {row['speaker_id']}: {row['utterances']} utts, {row['total_sec']/60:.1f} mins, {row['gender']}, {row['language']}, {row['state']}")

