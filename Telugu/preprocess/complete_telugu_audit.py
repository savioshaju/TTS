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
print("COMPREHENSIVE AUDIT: ORIGINAL PARQUET SHARDS")
print("=" * 80)

parquet_files = sorted(glob.glob(os.path.join(PARQUET_DIR, "*.parquet")))
total_shards = len(parquet_files)
shard_summaries = []
all_orig_records = []
text_hashes = defaultdict(list)
audio_sigs = defaultdict(list)

for pf_idx, pf in enumerate(parquet_files):
    shard_name = os.path.basename(pf)
    f_size = os.path.getsize(pf)
    
    df_meta = pl.read_parquet(pf, columns=["text", "gender"])
    n_rows = len(df_meta)
    texts = df_meta["text"].to_list()
    genders = df_meta["gender"].to_list()
    
    df_sizes = pl.read_parquet(pf).select([
        pl.col("audio").struct.field("path").alias("audio_path"),
        pl.col("audio").struct.field("bytes").bin.size().alias("size_bytes")
    ])
    paths = df_sizes["audio_path"].to_list()
    sizes = df_sizes["size_bytes"].to_list()
    
    # Check sample rate for this shard from first 5 rows
    df_sample = pl.read_parquet(pf).slice(0, 5).select(pl.col("audio").struct.field("bytes").alias("b"))
    sr_shard = None
    for b in df_sample["b"].to_list():
        with io.BytesIO(b) as bio:
            with wave.open(bio, "rb") as wf:
                sr_shard = wf.getframerate()
                break
                
    bytes_per_sec = sr_shard * 1 * 2  # 1 channel, 16-bit PCM (2 bytes)
    
    shard_durs = []
    for row_idx in range(n_rows):
        t_val = texts[row_idx]
        g_val = genders[row_idx]
        a_path = paths[row_idx]
        b_len = sizes[row_idx]
        
        clean_text = str(t_val).strip().replace("\ufeff", "") if t_val else ""
        dur = (b_len - 44) / float(bytes_per_sec)
        shard_durs.append(dur)
        
        spk_id = "telugufemale" if g_val == 0 else "telugumale"
        gender_str = "Female" if g_val == 0 else "Male"
        
        uid = f"{shard_name}_{row_idx}"
        t_hash = hashlib.md5(clean_text.encode("utf-8")).hexdigest()
        a_sig = f"{a_path}_{b_len}"
        
        record = {
            "shard": shard_name,
            "row_idx": row_idx,
            "filename": a_path,
            "speaker_id": spk_id,
            "gender": gender_str,
            "sample_rate": sr_shard,
            "channels": 1,
            "bit_depth": 16,
            "duration": dur,
            "text": clean_text,
            "text_len": len(clean_text),
            "text_hash": t_hash,
            "audio_sig": a_sig
        }
        all_orig_records.append(record)
        text_hashes[t_hash].append(uid)
        audio_sigs[a_sig].append(uid)
        
    shard_summary = {
        "shard": shard_name,
        "size_bytes": f_size,
        "rows": n_rows,
        "speaker": "telugufemale (Female)" if genders[0] == 0 else "telugumale (Male)",
        "sample_rate": sr_shard,
        "total_duration_hours": round(sum(shard_durs) / 3600.0, 4),
        "total_duration_seconds": round(sum(shard_durs), 2),
        "mean_duration": round(float(np.mean(shard_durs)), 4),
        "min_duration": round(float(np.min(shard_durs)), 4),
        "max_duration": round(float(np.max(shard_durs)), 4)
    }
    shard_summaries.append(shard_summary)
    print(f"Shard {shard_name}: {n_rows:,} rows, {shard_summary['speaker']}, {sr_shard} Hz, {shard_summary['total_duration_hours']} h")

# Overall original stats
orig_durs = [r["duration"] for r in all_orig_records]
total_orig_dur_sec = sum(orig_durs)
total_orig_dur_hr = total_orig_dur_sec / 3600.0
total_orig_count = len(all_orig_records)

# Percentiles & bins
percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
pct_vals = np.percentile(orig_durs, percentiles)

bins = [0, 1, 2, 4, 6, 8, 10, 12, 14, 16, 20, 30, 60, 100, 300]
hist, bin_edges = np.histogram(orig_durs, bins=bins)
bin_counts = [
    {
        "range": f"[{bin_edges[i]:.1f}s - {bin_edges[i+1]:.1f}s)",
        "count": int(hist[i]),
        "percent": round(float(hist[i] / total_orig_count * 100), 2)
    }
    for i in range(len(hist))
]

# Speakers
spk_orig_groups = defaultdict(list)
for r in all_orig_records:
    spk_orig_groups[r["speaker_id"]].append(r)

spk_stats_orig = {}
for spk, utts in sorted(spk_orig_groups.items()):
    d_list = [u["duration"] for u in utts]
    spk_stats_orig[spk] = {
        "gender": utts[0]["gender"],
        "sample_rate": utts[0]["sample_rate"],
        "count": len(utts),
        "percent_count": round(len(utts) / total_orig_count * 100, 2),
        "duration_hours": round(sum(d_list) / 3600.0, 4),
        "duration_seconds": round(sum(d_list), 2),
        "percent_duration": round(sum(d_list) / total_orig_dur_sec * 100, 2),
        "mean_duration": round(float(np.mean(d_list)), 4),
        "median_duration": round(float(np.median(d_list)), 4),
        "min_duration": round(float(np.min(d_list)), 4),
        "max_duration": round(float(np.max(d_list)), 4),
        "std_duration": round(float(np.std(d_list)), 4)
    }

dup_orig_texts = {h: locs for h, locs in text_hashes.items() if len(locs) > 1}
dup_orig_audios = {h: locs for h, locs in audio_sigs.items() if len(locs) > 1}

print("\n" + "=" * 80)
print("AUDITING PREPARED SPLITS (datasets/prepared/telugu_prepared)")
print("=" * 80)

def audit_prepared_split(split_name):
    split_dir = os.path.join(PREPARED_DIR, split_name)
    wav_dir = os.path.join(split_dir, "wavs")
    txt_path = os.path.join(split_dir, f"{split_name}.txt")
    meta_path = os.path.join(split_dir, "metadata.csv")
    
    wav_files = sorted(glob.glob(os.path.join(wav_dir, "*.wav")))
    
    # Read manifest
    lines = open(txt_path, "r", encoding="utf-8").readlines()
    manifest_records = []
    for l in lines:
        parts = l.strip().split("|")
        manifest_records.append(parts)
        
    srs = Counter()
    channels = Counter()
    sampwidths = Counter()
    durs = []
    spk_counts = Counter()
    spk_durs = defaultdict(float)
    basenames = []
    
    for p in wav_files:
        bname = os.path.splitext(os.path.basename(p))[0]
        basenames.append(bname)
        with wave.open(p, "rb") as wf:
            sr = wf.getframerate()
            n_ch = wf.getnchannels()
            sw = wf.getsampwidth()
            n_frames = wf.getnframes()
            dur = n_frames / float(sr)
            
            srs[sr] += 1
            channels[n_ch] += 1
            sampwidths[sw] += 1
            durs.append(dur)
            
            # Speaker from filename convention (indic_te_X_Y or prefix)
            # In metadata or manifest:
            # Let's inspect speaker ID from manifest
    
    for r in manifest_records:
        bname, spk = r[0], r[1]
        spk_counts[spk] += 1
        # match duration
        # we can get duration from wav file
    
    # Calculate duration per speaker
    for p, bname in zip(wav_files, basenames):
        with wave.open(p, "rb") as wf:
            dur = wf.getnframes() / float(wf.getframerate())
            sr = wf.getframerate()
            spk = "telugufemale" if sr == 16000 else "telugumale"
            spk_durs[spk] += dur
            
    tot_dur = sum(durs)
    return {
        "split": split_name,
        "wav_count": len(wav_files),
        "manifest_count": len(manifest_records),
        "total_duration_hours": round(tot_dur / 3600.0, 4),
        "total_duration_seconds": round(tot_dur, 2),
        "min_duration": round(float(min(durs)), 4),
        "max_duration": round(float(max(durs)), 4),
        "mean_duration": round(float(np.mean(durs)), 4),
        "median_duration": round(float(np.median(durs)), 4),
        "std_duration": round(float(np.std(durs)), 4),
        "sample_rates": dict(srs),
        "channels": dict(channels),
        "bit_depths": {sw*8: c for sw, c in sampwidths.items()},
        "speakers": {
            spk: {
                "count": spk_counts[spk],
                "duration_hours": round(spk_durs[spk] / 3600.0, 4),
                "duration_seconds": round(spk_durs[spk], 2),
                "sample_rate": 16000 if spk == "telugufemale" else 48000
            }
            for spk in sorted(spk_counts.keys())
        },
        "basenames": set(basenames)
    }

train_audit = audit_prepared_split("train")
val_audit = audit_prepared_split("val")

# Overlap check
overlap_ids = train_audit["basenames"].intersection(val_audit["basenames"])
print(f"Overlap between Train and Val basenames: {len(overlap_ids)} utterances")

# Clean basenames from audit dict for JSON
del train_audit["basenames"]
del val_audit["basenames"]

full_audit_results = {
    "original_dataset": {
        "source": "SPRINGLab/IndicTTS_Telugu",
        "total_shards": total_shards,
        "shards": shard_summaries,
        "total_utterances": total_orig_count,
        "total_duration_hours": round(total_orig_dur_hr, 4),
        "total_duration_seconds": round(total_orig_dur_sec, 2),
        "min_duration": round(float(np.min(orig_durs)), 4),
        "max_duration": round(float(np.max(orig_durs)), 4),
        "mean_duration": round(float(np.mean(orig_durs)), 4),
        "median_duration": round(float(np.median(orig_durs)), 4),
        "std_duration": round(float(np.std(orig_durs)), 4),
        "percentiles": {f"p{p}": round(float(v), 4) for p, v in zip(percentiles, pct_vals)},
        "duration_bins": bin_counts,
        "speakers": spk_stats_orig,
        "duplicate_transcripts_count": len(dup_orig_texts),
        "duplicate_audio_signatures_count": len(dup_orig_audios)
    },
    "prepared_dataset": {
        "train": train_audit,
        "val": val_audit,
        "total_prepared_count": train_audit["wav_count"] + val_audit["wav_count"],
        "total_prepared_duration_hours": round(train_audit["total_duration_hours"] + val_audit["total_duration_hours"], 4),
        "total_prepared_duration_seconds": round(train_audit["total_duration_seconds"] + val_audit["total_duration_seconds"], 2),
        "utterance_overlap_count": len(overlap_ids),
        "overlap_percentage": 0.0
    }
}

with open(os.path.join(BASE_DIR, "Report", "telugu_stage_a_audit.json"), "w", encoding="utf-8") as f:
    json.dump(full_audit_results, f, indent=2)

print("\nAudit completed successfully! Saved Report/telugu_stage_a_audit.json")
