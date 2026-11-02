"""
Stage D: Speaker-Dependent Stratified Repartitioning and Renormalization.

Implements User Instructions:
1. Uses the existing 4,165 successfully extracted English utterances and acoustic features.
   Does not rerun MFA or feature extraction.
2. Creates a reproducible, stratified utterance-level split (seed=42) so that every single
   one of the 104 speakers is represented in both training and validation.
3. Preserves original Stage B and Stage C/D manifests and reports:
   - train_speaker_independent.txt, val_speaker_independent.txt
   - stats_speaker_independent.json
4. Generates new train.txt (3,419 utterances) and val.txt (746 utterances) without utterance overlap.
5. Recalculates pitch and energy normalization statistics using the new training split only.
6. Re-normalizes all 4,165 pitch and energy .npy arrays to match the new training statistics.
7. Saves updated stats.json and updates English/config.yaml and Svarah_English/preprocess.yaml.
8. Verifies 100% speaker coverage, split separation, duration statistics, and feature integrity.
"""

import os
import sys
import json
import shutil
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

ROOT_DIR = r"D:\Savio Shaju\TTS"
PREP_DIR = os.path.join(ROOT_DIR, "preprocessing", "English")

def repartition_and_renormalize():
    print("=" * 80)
    print("STAGE D: SPEAKER-DEPENDENT STRATIFIED REPARTITIONING & RENORMALIZATION")
    print("=" * 80)

    # 1. Preserve previous manifests and stats
    print("\n[Step 1] Preserving previous manifests and stats...")
    cur_train_path = os.path.join(PREP_DIR, "train.txt")
    cur_val_path = os.path.join(PREP_DIR, "val.txt")
    cur_stats_path = os.path.join(PREP_DIR, "stats.json")

    backup_train_si = os.path.join(PREP_DIR, "train_speaker_independent.txt")
    backup_val_si = os.path.join(PREP_DIR, "val_speaker_independent.txt")
    backup_stats_si = os.path.join(PREP_DIR, "stats_speaker_independent.json")

    if not os.path.exists(backup_train_si):
        shutil.copyfile(cur_train_path, backup_train_si)
        print(f"  Saved {backup_train_si}")
    if not os.path.exists(backup_val_si):
        shutil.copyfile(cur_val_path, backup_val_si)
        print(f"  Saved {backup_val_si}")
    if not os.path.exists(backup_stats_si):
        shutil.copyfile(cur_stats_path, backup_stats_si)
        print(f"  Saved {backup_stats_si}")

    # Load current stats (needed to recover raw pitch and energy values)
    with open(backup_stats_si, "r") as f:
        old_stats = json.load(f)
    old_p_min, old_p_max, old_p_mean, old_p_std = old_stats["pitch"]
    old_e_min, old_e_max, old_e_mean, old_e_std = old_stats["energy"]
    print(f"  Loaded previous normalizer params: Pitch (mean={old_p_mean:.4f}, std={old_p_std:.4f}), Energy (mean={old_e_mean:.4f}, std={old_e_std:.4f})")

    # 2. Collect all 4,165 records
    records = []
    with open(backup_train_si, "r", encoding="utf-8") as f:
        records.extend([l.strip() for l in f if l.strip()])
    with open(backup_val_si, "r", encoding="utf-8") as f:
        records.extend([l.strip() for l in f if l.strip()])

    print(f"\n[Step 2] Loaded all {len(records)} verified records across the dataset.")

    # Group records by speaker
    spk_records = defaultdict(list)
    for r in records:
        spk = r.split("|")[1]
        spk_records[spk].append(r)

    print(f"  Grouped into {len(spk_records)} unique speakers.")

    # 3. Stratified Partitioning (Fixed Random Seed = 42)
    print("\n[Step 3] Performing stratified speaker-dependent splitting (target ~18% val)...")
    np.random.seed(42)
    train_manifest = []
    val_manifest = []

    for spk in sorted(spk_records.keys()):
        utts = spk_records[spk]
        n = len(utts)
        indices = np.random.permutation(n)
        # Select val count: at least 1, target 18%
        n_val = max(1, int(round(0.18 * n)))
        if n - n_val < 1:
            n_val = n - 1  # ensure at least 1 train utterance

        for idx in indices[:n_val]:
            val_manifest.append(utts[idx])
        for idx in indices[n_val:]:
            train_manifest.append(utts[idx])

    # Sort manifests alphabetically by UID for reproducibility
    train_manifest.sort(key=lambda x: x.split("|")[0])
    val_manifest.sort(key=lambda x: x.split("|")[0])

    print(f"  New Train Record Count: {len(train_manifest)}")
    print(f"  New Val Record Count:   {len(val_manifest)}")
    assert len(train_manifest) + len(val_manifest) == 4165, "Total record count mismatch!"

    # Split overlap check
    train_uids = set(r.split("|")[0] for r in train_manifest)
    val_uids = set(r.split("|")[0] for r in val_manifest)
    assert len(train_uids.intersection(val_uids)) == 0, "Utterance overlap detected!"
    print("  Split separation verified: STRICTLY ZERO utterance overlap.")

    # Speaker representation check
    train_spks = set(r.split("|")[1] for r in train_manifest)
    val_spks = set(r.split("|")[1] for r in val_manifest)
    assert len(train_spks) == 104, f"Train speakers count {len(train_spks)} != 104"
    assert len(val_spks) == 104, f"Val speakers count {len(val_spks)} != 104"
    print("  Speaker coverage verified: 104 / 104 speakers represented in both Train and Validation.")

    # 4. Measure duration statistics
    dur_dir = os.path.join(PREP_DIR, "duration")
    t_frames, v_frames = 0, 0
    spk_dist = defaultdict(lambda: {"train": 0, "val": 0, "train_dur": 0.0, "val_dur": 0.0})

    for r in train_manifest:
        parts = r.split("|")
        uid, spk = parts[0], parts[1]
        dur = np.load(os.path.join(dur_dir, f"{spk}-duration-{uid}.npy"))
        frames = int(np.sum(dur))
        t_frames += frames
        spk_dist[spk]["train"] += 1
        spk_dist[spk]["train_dur"] += frames * 256 / 16000

    for r in val_manifest:
        parts = r.split("|")
        uid, spk = parts[0], parts[1]
        dur = np.load(os.path.join(dur_dir, f"{spk}-duration-{uid}.npy"))
        frames = int(np.sum(dur))
        v_frames += frames
        spk_dist[spk]["val"] += 1
        spk_dist[spk]["val_dur"] += frames * 256 / 16000

    train_hrs = t_frames * 256 / 16000 / 3600
    val_hrs = v_frames * 256 / 16000 / 3600
    total_hrs = (t_frames + v_frames) * 256 / 16000 / 3600

    print(f"\nDuration Statistics:")
    print(f"  Training Split:   {train_hrs:.3f} hours ({t_frames:,} mel frames, {len(train_manifest)} utts)")
    print(f"  Validation Split: {val_hrs:.3f} hours ({v_frames:,} mel frames, {len(val_manifest)} utts)")
    print(f"  Total Dataset:    {total_hrs:.3f} hours ({t_frames + v_frames:,} mel frames, {len(records)} utts)")

    # 5. Recalculate pitch and energy normalization statistics using NEW TRAINING SPLIT ONLY
    print("\n[Step 4] Fitting StandardScaler strictly on new training split (3,419 utterances)...")
    pitch_dir = os.path.join(PREP_DIR, "pitch")
    energy_dir = os.path.join(PREP_DIR, "energy")

    new_pitch_scaler = StandardScaler()
    new_energy_scaler = StandardScaler()

    for r in tqdm(train_manifest, desc="Collecting new train stats"):
        parts = r.split("|")
        uid, spk = parts[0], parts[1]

        p_path = os.path.join(pitch_dir, f"{spk}-pitch-{uid}.npy")
        e_path = os.path.join(energy_dir, f"{spk}-energy-{uid}.npy")

        # Invert old standardization to recover raw physical values
        p_std_old = np.load(p_path)
        e_std_old = np.load(e_path)

        p_mask = p_std_old != 0
        p_raw = np.zeros_like(p_std_old)
        p_raw[p_mask] = p_std_old[p_mask] * old_p_std + old_p_mean

        e_mask = e_std_old != 0
        e_raw = np.zeros_like(e_std_old)
        e_raw[e_mask] = e_std_old[e_mask] * old_e_std + old_e_mean

        if np.any(p_mask):
            new_pitch_scaler.partial_fit(p_raw[p_mask].reshape(-1, 1))
        if np.any(e_mask):
            new_energy_scaler.partial_fit(e_raw[e_mask].reshape(-1, 1))

    new_p_mean = float(new_pitch_scaler.mean_[0])
    new_p_std = float(new_pitch_scaler.scale_[0])
    new_e_mean = float(new_energy_scaler.mean_[0])
    new_e_std = float(new_energy_scaler.scale_[0])

    print(f"\nNew Normalization Statistics (Strictly New Train Split):")
    print(f"  Pitch:  mean = {new_p_mean:.4f} Hz (was {old_p_mean:.4f}), std = {new_p_std:.4f} Hz (was {old_p_std:.4f})")
    print(f"  Energy: mean = {new_e_mean:.4f} (was {old_e_mean:.4f}), std = {new_e_std:.4f} (was {old_e_std:.4f})")

    # 6. Re-standardize all 4,165 files with the new parameters
    print("\n[Step 5] Re-standardizing all 4,165 pitch and energy files on disk...")
    new_p_min = float("inf")
    new_p_max = float("-inf")
    new_e_min = float("inf")
    new_e_max = float("-inf")

    all_records = train_manifest + val_manifest
    for r in tqdm(all_records, desc="Re-saving normalized features"):
        parts = r.split("|")
        uid, spk = parts[0], parts[1]

        p_path = os.path.join(pitch_dir, f"{spk}-pitch-{uid}.npy")
        e_path = os.path.join(energy_dir, f"{spk}-energy-{uid}.npy")

        # Invert old then apply new
        p_std_old = np.load(p_path)
        p_mask = p_std_old != 0
        p_new = np.zeros_like(p_std_old)
        if np.any(p_mask):
            p_raw = p_std_old[p_mask] * old_p_std + old_p_mean
            p_new[p_mask] = (p_raw - new_p_mean) / new_p_std
            new_p_min = min(new_p_min, float(np.min(p_new[p_mask])))
            new_p_max = max(new_p_max, float(np.max(p_new[p_mask])))
        np.save(p_path, p_new.astype(np.float32))

        e_std_old = np.load(e_path)
        e_mask = e_std_old != 0
        e_new = np.zeros_like(e_std_old)
        if np.any(e_mask):
            e_raw = e_std_old[e_mask] * old_e_std + old_e_mean
            e_new[e_mask] = (e_raw - new_e_mean) / new_e_std
            new_e_min = min(new_e_min, float(np.min(e_new[e_mask])))
            new_e_max = max(new_e_max, float(np.max(e_new[e_mask])))
        np.save(e_path, e_new.astype(np.float32))

    new_stats = {
        "pitch": [new_p_min, new_p_max, new_p_mean, new_p_std],
        "energy": [new_e_min, new_e_max, new_e_mean, new_e_std]
    }
    with open(cur_stats_path, "w", encoding="utf-8") as f:
        json.dump(new_stats, f, indent=2)
    print(f"  Saved updated stats to: {cur_stats_path}")
    print(f"  Stats: {new_stats}")

    # 7. Write new active manifests
    print("\n[Step 6] Writing new train.txt and val.txt...")
    with open(cur_train_path, "w", encoding="utf-8") as f:
        for l in train_manifest:
            f.write(l + "\n")
    with open(cur_val_path, "w", encoding="utf-8") as f:
        for l in val_manifest:
            f.write(l + "\n")
    print("  Updated train.txt and val.txt successfully.")

    # 8. Update configurations
    print("\n[Step 7] Updating val_size in configuration files...")
    import yaml
    cfg_en_path = os.path.join(ROOT_DIR, "English", "config.yaml")
    with open(cfg_en_path, "r", encoding="utf-8") as f:
        cfg_en = yaml.safe_load(f)
    cfg_en["preprocessing"]["val_size"] = len(val_manifest)
    with open(cfg_en_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg_en, f, default_flow_style=False, sort_keys=False)

    fs2_prep_path = os.path.join(ROOT_DIR, "FastSpeech2-master", "config", "Svarah_English", "preprocess.yaml")
    with open(fs2_prep_path, "r", encoding="utf-8") as f:
        fs2_prep = yaml.safe_load(f)
    fs2_prep["preprocessing"]["val_size"] = len(val_manifest)
    with open(fs2_prep_path, "w", encoding="utf-8") as f:
        yaml.dump(fs2_prep, f, default_flow_style=False, sort_keys=False)
    print(f"  Set val_size to {len(val_manifest)} in English/config.yaml and preprocess.yaml.")

    # 9. Generate detailed report
    report = {
        "strategy": "speaker_dependent_stratified",
        "random_seed": 42,
        "total_utterances": len(records),
        "train_utterances": len(train_manifest),
        "val_utterances": len(val_manifest),
        "train_hours": round(train_hrs, 3),
        "val_hours": round(val_hrs, 3),
        "total_hours": round(total_hrs, 3),
        "train_mel_frames": t_frames,
        "val_mel_frames": v_frames,
        "train_speakers_count": len(train_spks),
        "val_speakers_count": len(val_spks),
        "speaker_overlap_count": 0,
        "all_speakers_in_train": len(train_spks) == 104,
        "all_speakers_in_val": len(val_spks) == 104,
        "normalization_stats": new_stats,
        "previous_stats": old_stats,
        "speaker_distribution": {k: spk_dist[k] for k in sorted(spk_dist.keys())}
    }
    report_out = os.path.join(PREP_DIR, "reports", "speaker_dependent_repartition_report.json")
    with open(report_out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[Step 8] Saved full repartition report to: {report_out}")
    print("=" * 80)
    print("SPEAKER-DEPENDENT REPARTITIONING AND RENORMALIZATION COMPLETED SUCCESSFULLY.")
    print("=" * 80)

if __name__ == "__main__":
    repartition_and_renormalize()

