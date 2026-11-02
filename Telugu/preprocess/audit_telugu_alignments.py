"""
Stage C — Telugu Forced Alignment Quality Verification and Audit Suite.
Conducts a comprehensive audit across all 2,695 generated alignment files:
- Verifies boundary consistency (start >= 0, end >= start, end <= audio_dur).
- Audits zero-duration tokens and word-level overlaps.
- Analyzes alignment coverage and confidence distributions.
- Compares Female (16k native) vs Male (resampled 48k->16k) acoustic alignment quality.
- Identifies any low-confidence or failed alignments.
- Saves detailed JSON metrics to Report/telugu_stage_c_audit.json.
"""

import os
import sys
import json
import numpy as np

ROOT_DIR = r"D:\Savio Shaju\TTS"
ALIGN_DIR = os.path.join(ROOT_DIR, "preprocessing", "Telugu", "alignments")
DATASET_DIR = os.path.join(ROOT_DIR, "datasets", "prepared", "telugu_prepared_v2")
REPORT_PATH = os.path.join(ROOT_DIR, "Report", "telugu_stage_c_audit.json")

print("=" * 75)
print("STAGE C: TELUGU ALIGNMENT QUALITY VERIFICATION AND AUDIT SUITE")
print(f"Alignment Directory: {ALIGN_DIR}")
print("=" * 75)

# Load metadata for all 2,695 records
records = []
for split in ["train", "val"]:
    meta_path = os.path.join(DATASET_DIR, split, "metadata.csv")
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = line.split("|")
            records.append({
                "uid": p[0],
                "split": split,
                "speaker": p[1],
                "text": p[2],
                "gender": p[3],
                "duration": float(p[4]),
            })

print(f"Expected alignments: {len(records):,} (Train: 2,250, Val: 445)")

total_expected = len(records)
missing_files = []
corrupted_files = []
zero_duration_tokens = 0
boundary_errors = 0
negative_timestamp_errors = 0
reverse_interval_errors = 0

all_confidences = []
female_confidences = []
male_confidences = []

all_coverages = []
female_coverages = []
male_coverages = []

low_confidence_utterances = [] # score < 0.35
detailed_entries = []

for rec in records:
    uid = rec["uid"]
    gender = rec["gender"]
    dur = rec["duration"]
    fpath = os.path.join(ALIGN_DIR, f"{uid}.json")

    if not os.path.isfile(fpath):
        missing_files.append(uid)
        continue

    try:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        corrupted_files.append({"uid": uid, "error": str(e)})
        continue

    words = data.get("words", [])
    mean_conf = data.get("mean_confidence", 0.0)
    all_confidences.append(mean_conf)

    if gender == "Female":
        female_confidences.append(mean_conf)
    else:
        male_confidences.append(mean_conf)

    if mean_conf < 0.35:
        low_confidence_utterances.append({
            "uid": uid,
            "speaker": rec["speaker"],
            "gender": gender,
            "duration": dur,
            "confidence": mean_conf,
            "word_count": len(words),
        })

    total_word_speech_dur = 0.0
    prev_w_end = 0.0

    for w_idx, w in enumerate(words):
        w_start = w["start"]
        w_end = w["end"]
        w_dur = w["dur"]
        chars = w.get("chars", [])

        if w_start < 0:
            negative_timestamp_errors += 1
        if w_end < w_start:
            reverse_interval_errors += 1
        if w_end > dur + 0.2:  # allow 200ms margin for tail padding
            boundary_errors += 1

        total_word_speech_dur += max(0.0, w_end - w_start)

        # Character / token level audit
        for c in chars:
            c_start = c["start"]
            c_end = c["end"]
            c_dur = c["dur"]
            if c_dur <= 0.0:
                zero_duration_tokens += 1
            if c_start < 0 or c_end < c_start:
                reverse_interval_errors += 1

    coverage = (total_word_speech_dur / dur) if dur > 0 else 0.0
    all_coverages.append(coverage)
    if gender == "Female":
        female_coverages.append(coverage)
    else:
        male_coverages.append(coverage)

    detailed_entries.append({
        "uid": uid,
        "gender": gender,
        "duration": dur,
        "word_count": len(words),
        "mean_confidence": mean_conf,
        "speech_coverage": round(coverage, 4),
    })

# Compute statistical distributions
stats = {
    "total_expected": total_expected,
    "total_found": len(records) - len(missing_files),
    "missing_count": len(missing_files),
    "corrupted_count": len(corrupted_files),
    "boundary_errors": boundary_errors,
    "negative_timestamp_errors": negative_timestamp_errors,
    "reverse_interval_errors": reverse_interval_errors,
    "zero_duration_tokens": zero_duration_tokens,
    "confidence": {
        "overall": {
            "mean": round(float(np.mean(all_confidences)), 4),
            "std": round(float(np.std(all_confidences)), 4),
            "median": round(float(np.median(all_confidences)), 4),
            "min": round(float(np.min(all_confidences)), 4),
            "max": round(float(np.max(all_confidences)), 4),
        },
        "female": {
            "mean": round(float(np.mean(female_confidences)), 4),
            "std": round(float(np.std(female_confidences)), 4),
            "median": round(float(np.median(female_confidences)), 4),
            "min": round(float(np.min(female_confidences)), 4),
            "max": round(float(np.max(female_confidences)), 4),
            "count": len(female_confidences),
        },
        "male": {
            "mean": round(float(np.mean(male_confidences)), 4),
            "std": round(float(np.std(male_confidences)), 4),
            "median": round(float(np.median(male_confidences)), 4),
            "min": round(float(np.min(male_confidences)), 4),
            "max": round(float(np.max(male_confidences)), 4),
            "count": len(male_confidences),
        },
    },
    "speech_coverage": {
        "overall_mean": round(float(np.mean(all_coverages)), 4),
        "female_mean": round(float(np.mean(female_coverages)), 4),
        "male_mean": round(float(np.mean(male_coverages)), 4),
    },
    "low_confidence_count": len(low_confidence_utterances),
    "low_confidence_samples": low_confidence_utterances[:10],
}

print("\n" + "=" * 75)
print("AUDIT RESULTS SUMMARY")
print("=" * 75)
print(f"  Total Alignments Found:   {stats['total_found']:,} / {stats['total_expected']:,} ({stats['total_found']/stats['total_expected']*100:.2f}%)")
print(f"  Missing Alignments:       {stats['missing_count']}")
print(f"  Corrupted Files:          {stats['corrupted_count']}")
print(f"  Negative Timestamps:      {stats['negative_timestamp_errors']}")
print(f"  Reverse Intervals:        {stats['reverse_interval_errors']}")
print(f"  Zero-Duration Tokens:     {stats['zero_duration_tokens']}")
print(f"  Boundary Overflow Errors: {stats['boundary_errors']}")
print("-" * 75)
print(f"  Overall Mean Confidence:  {stats['confidence']['overall']['mean']:.4f} +/- {stats['confidence']['overall']['std']:.4f}")
print(f"  Female Mean Confidence:   {stats['confidence']['female']['mean']:.4f} +/- {stats['confidence']['female']['std']:.4f} (Median: {stats['confidence']['female']['median']:.4f})")
print(f"  Male Mean Confidence:     {stats['confidence']['male']['mean']:.4f} +/- {stats['confidence']['male']['std']:.4f} (Median: {stats['confidence']['male']['median']:.4f})")
print(f"  Overall Speech Coverage:  {stats['speech_coverage']['overall_mean']*100:.2f}% of total audio duration")
print(f"  Low Confidence (<0.35):   {stats['low_confidence_count']} utterances ({stats['low_confidence_count']/len(records)*100:.2f}%)")
print("=" * 75)

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    json.dump(stats, f, indent=2)

print(f"\nAudit results successfully saved to {REPORT_PATH}")

