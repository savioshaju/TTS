import os
import io
import glob
import wave
import json
import hashlib
import numpy as np
import polars as pl
from collections import Counter, defaultdict

BASE_DIR = r"D:\Savio Shaju\TTS"
PARQUET_DIR = os.path.join(BASE_DIR, "datasets", "IndicTTS_Telugu", "data")
PREPARED_DIR = os.path.join(BASE_DIR, "datasets", "prepared", "telugu_prepared")

print("=" * 80)
print("AUDITING ORIGINAL TELUGU PARQUET DATASET (IndicTTS_Telugu)")
print("=" * 80)

parquet_files = sorted(glob.glob(os.path.join(PARQUET_DIR, "*.parquet")))
print(f"Found {len(parquet_files)} Parquet shards:")
for pf in parquet_files:
    print(f"  {os.path.basename(pf)}: {os.path.getsize(pf):,} bytes")

all_records = []
text_hashes = defaultdict(list)
audio_hashes = defaultdict(list)
sample_rates = Counter()
channels_counter = Counter()
sampwidths = Counter()
empty_transcripts = 0
corrupted_audio = 0

total_shards = len(parquet_files)

for pf_idx, pf in enumerate(parquet_files):
    shard_name = os.path.basename(pf)
    print(f"\nScanning shard {pf_idx + 1}/{total_shards}: {shard_name}...")
    
    # 1. Read metadata: text and gender
    df_meta = pl.read_parquet(pf, columns=["text", "gender"])
    n_rows = len(df_meta)
    print(f"  Rows: {n_rows:,}")
    
    texts = df_meta["text"].to_list()
    genders = df_meta["gender"].to_list()
    
    # 2. Vectorized extraction of audio size and path without copying all raw bytes to Python heap
    df_sizes = pl.read_parquet(pf).select([
        pl.col("audio").struct.field("path").alias("audio_path"),
        pl.col("audio").struct.field("bytes").bin.size().alias("size_bytes")
    ])
    paths = df_sizes["audio_path"].to_list()
    sizes = df_sizes["size_bytes"].to_list()
    
    # 3. Read a single small slice (20 rows) of audio to inspect wave headers
    df_sample_audio = pl.read_parquet(pf).slice(0, 20).select(pl.col("audio").struct.field("bytes").alias("b"))
    sample_bytes_list = df_sample_audio["b"].to_list()
    for raw_b in sample_bytes_list:
        try:
            with io.BytesIO(raw_b) as bio:
                with wave.open(bio, "rb") as wf:
                    sr = wf.getframerate()
                    n_ch = wf.getnchannels()
                    sw = wf.getsampwidth()
                    sample_rates[sr] += 1
                    channels_counter[n_ch] += 1
                    sampwidths[sw] += 1
        except Exception as e:
            corrupted_audio += 1
    
    for row_idx in range(n_rows):
        t_val = texts[row_idx]
        g_val = genders[row_idx]
        a_path = paths[row_idx]
        b_len = sizes[row_idx]
        
        if t_val is None or len(str(t_val).strip()) == 0:
            empty_transcripts += 1
            clean_text = ""
        else:
            clean_text = str(t_val).strip().replace("\ufeff", "")
            
        if b_len is None or b_len < 44:
            corrupted_audio += 1
            continue
            
        # 16-bit PCM WAV: 16000 Hz, 1 channel, 2 bytes/sample -> 32,000 bytes/sec
        dur = (b_len - 44) / 32000.0
        
        t_hash = hashlib.md5(clean_text.encode("utf-8")).hexdigest()
        # Audio signature using filename + size
        a_sig = f"{a_path}_{b_len}"
        
        spk_id = "telugufemale" if g_val == 0 else "telugumale"
        gender_str = "Female" if g_val == 0 else "Male"
        
        uid = f"{shard_name}_{row_idx}"
        record = {
            "shard": shard_name,
            "row_idx": row_idx,
            "filename": a_path,
            "speaker_id": spk_id,
            "gender": gender_str,
            "gender_code": g_val,
            "size_bytes": b_len,
            "duration": dur,
            "text": clean_text,
            "text_len": len(clean_text),
            "text_hash": t_hash,
            "audio_sig": a_sig
        }
        all_records.append(record)
        text_hashes[t_hash].append(uid)
        audio_hashes[a_sig].append(uid)

print("\n" + "=" * 80)
print("ORIGINAL TELUGU DATASET AUDIT SUMMARY")
print("=" * 80)

total_count = len(all_records)
durations = [r["duration"] for r in all_records]
total_dur_sec = sum(durations)
total_dur_hr = total_dur_sec / 3600.0

print(f"Total Utterances:            {total_count:,}")
print(f"Total Audio Duration:        {total_dur_hr:.4f} hours ({total_dur_sec:,.2f} seconds)")
print(f"Minimum Duration:            {np.min(durations):.4f} seconds")
print(f"Maximum Duration:            {np.max(durations):.4f} seconds")
print(f"Mean Duration:               {np.mean(durations):.4f} seconds")
print(f"Median Duration:             {np.median(durations):.4f} seconds")
print(f"Std Deviation Duration:      {np.std(durations):.4f} seconds")

# Percentiles
percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
pct_vals = np.percentile(durations, percentiles)
print("\nDuration Percentiles:")
for p, v in zip(percentiles, pct_vals):
    print(f"  {p:2d}th percentile:            {v:.4f} s")

# Duration distribution bins
bins = [0, 1, 2, 4, 6, 8, 10, 12, 14, 16, 20, 30, 100]
hist, bin_edges = np.histogram(durations, bins=bins)
print("\nDuration Distribution Bins:")
for i in range(len(hist)):
    print(f"  [{bin_edges[i]:4.1f}s - {bin_edges[i+1]:4.1f}s): {hist[i]:5d} utterances ({hist[i]/total_count*100:5.2f}%)")

print(f"\nAudio Sample Rates (sampled): {dict(sample_rates)}")
print(f"Audio Channels (sampled):     {dict(channels_counter)}")
print(f"Audio Bit Depths (sampled):   {{{k*8: v for k, v in sampwidths.items()}}}")
print(f"Corrupted Audio Records:      {corrupted_audio}")
print(f"Empty Transcripts:            {empty_transcripts}")

# Speaker breakdown
spk_groups = defaultdict(list)
for r in all_records:
    spk_groups[r["speaker_id"]].append(r)

print(f"\nUnique Speakers: {len(spk_groups)}")
speaker_summary = {}
for spk, utts in sorted(spk_groups.items()):
    spk_durs = [u["duration"] for u in utts]
    spk_dur_hr = sum(spk_durs) / 3600.0
    g = utts[0]["gender"]
    print(f"\n  Speaker: {spk} ({g})")
    print(f"    Utterance Count:         {len(utts):,} ({len(utts)/total_count*100:.2f}%)")
    print(f"    Total Audio Duration:    {spk_dur_hr:.4f} hours ({sum(spk_durs):,.2f} s)")
    print(f"    Mean Utterance Duration: {np.mean(spk_durs):.4f} s")
    print(f"    Median Duration:         {np.median(spk_durs):.4f} s")
    print(f"    Min / Max Duration:      {np.min(spk_durs):.4f} s / {np.max(spk_durs):.4f} s")
    
    speaker_summary[spk] = {
        "gender": g,
        "count": len(utts),
        "duration_hours": round(spk_dur_hr, 4),
        "duration_seconds": round(sum(spk_durs), 2),
        "mean_duration": round(float(np.mean(spk_durs)), 4),
        "median_duration": round(float(np.median(spk_durs)), 4),
        "min_duration": round(float(np.min(spk_durs)), 4),
        "max_duration": round(float(np.max(spk_durs)), 4),
    }

# Duplicate detection
dup_texts = {h: locs for h, locs in text_hashes.items() if len(locs) > 1}
dup_audios = {h: locs for h, locs in audio_hashes.items() if len(locs) > 1}
print(f"\nDuplicate Transcript Sets:    {len(dup_texts):,} (spanning {sum(len(l) for l in dup_texts.values()):,} utterances)")
print(f"Duplicate Audio Filename/Size: {len(dup_audios):,} (spanning {sum(len(l) for l in dup_audios.values()):,} utterances)")

# Save original stats to JSON
orig_stats = {
    "total_utterances": total_count,
    "total_duration_hours": round(total_dur_hr, 4),
    "total_duration_seconds": round(total_dur_sec, 2),
    "min_duration": round(float(np.min(durations)), 4),
    "max_duration": round(float(np.max(durations)), 4),
    "mean_duration": round(float(np.mean(durations)), 4),
    "median_duration": round(float(np.median(durations)), 4),
    "std_duration": round(float(np.std(durations)), 4),
    "percentiles": {f"p{p}": round(float(v), 4) for p, v in zip(percentiles, pct_vals)},
    "sample_rate": dict(sample_rates),
    "channels": dict(channels_counter),
    "bit_depth": {k*8: v for k, v in sampwidths.items()},
    "speakers": speaker_summary,
    "duplicate_transcript_sets": len(dup_texts),
    "duplicate_audio_sets": len(dup_audios),
    "corrupted_audio": corrupted_audio,
    "empty_transcripts": empty_transcripts
}

os.makedirs(os.path.join(BASE_DIR, "Report"), exist_ok=True)
with open(os.path.join(BASE_DIR, "Report", "telugu_original_stats.json"), "w", encoding="utf-8") as f:
    json.dump(orig_stats, f, indent=2)

print("\nSaved original stats to Report/telugu_original_stats.json.")
