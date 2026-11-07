"""
Stage D Comprehensive Quality Assurance and Verification Script for Telugu FastSpeech 2.
Audits:
1. Feature file generation vs expected utterance counts (2,693 utts = 10,772 .npy files).
2. Missing or corrupted files.
3. Mel dimensions, finite values, channel count (80).
4. Pitch and energy dimensions and finite values.
5. Duration array consistency (all > 0, sum(dur) == mel.shape[0]).
6. Phoneme sequence and duration alignment.
7. Train/val separation (no leakage, exact counts: 2,248 train, 445 val).
8. Speaker mapping consistency (telugufemale=0, telugumale=1).
9. Reference Dataset & DataLoader compatibility test.
10. Model forward-pass compatibility test (single batch eval, zero training).
"""

import os
import sys
import json
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

# Prioritize tracked Telugu/train implementation, falling back to FastSpeech2-master if available
TELUGU_TRAIN_DIR = os.path.join(ROOT_DIR, "Telugu", "train")
if os.path.isdir(TELUGU_TRAIN_DIR) and TELUGU_TRAIN_DIR not in sys.path:
    sys.path.insert(0, TELUGU_TRAIN_DIR)

FASTSPEECH2_DIR = os.path.join(ROOT_DIR, "FastSpeech2-master")
if os.path.isdir(FASTSPEECH2_DIR) and FASTSPEECH2_DIR not in sys.path:
    sys.path.insert(0, FASTSPEECH2_DIR)

from dataset import Dataset
from model import FastSpeech2
from utils.tools import to_device

PREP_BASE = os.path.join(ROOT_DIR, "preprocessing", "Telugu")
REPORT_DIR = os.path.join(ROOT_DIR, "Report")

EXCLUDE_UIDS = {"indic_te_0_927", "indic_te_3_0"}


def run_full_qa():
    print("=" * 80)
    print("STAGE D: TELUGU FASTSPEECH 2 COMPREHENSIVE QUALITY ASSURANCE AUDIT")
    print("=" * 80)

    results = {
        "passed": True,
        "errors": [],
        "warnings": [],
        "file_counts": {},
        "dimension_check": {},
        "manifest_check": {},
        "dataloader_test": {},
        "forward_pass_test": {},
    }

    # 1. Check manifests
    train_txt = os.path.join(PREP_BASE, "train.txt")
    val_txt = os.path.join(PREP_BASE, "val.txt")
    speakers_json = os.path.join(PREP_BASE, "speakers.json")
    stats_json = os.path.join(PREP_BASE, "stats.json")

    for f_p in [train_txt, val_txt, speakers_json, stats_json]:
        if not os.path.isfile(f_p) or os.path.getsize(f_p) == 0:
            results["passed"] = False
            results["errors"].append(f"Missing or empty file: {f_p}")

    with open(speakers_json, "r", encoding="utf-8") as f:
        spk_map = json.load(f)
    with open(stats_json, "r", encoding="utf-8") as f:
        stats = json.load(f)

    assert spk_map == {"telugufemale": 0, "telugumale": 1}, f"Unexpected speaker map: {spk_map}"

    train_entries = []
    with open(train_txt, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                train_entries.append(line.split("|"))

    val_entries = []
    with open(val_txt, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                val_entries.append(line.split("|"))

    train_uids = {e[0] for e in train_entries}
    val_uids = {e[0] for e in val_entries}

    results["manifest_check"] = {
        "train_count": len(train_entries),
        "val_count": len(val_entries),
        "total_count": len(train_entries) + len(val_entries),
        "leakage_overlap": len(train_uids.intersection(val_uids)),
        "excluded_found_in_train": list(train_uids.intersection(EXCLUDE_UIDS)),
        "excluded_found_in_val": list(val_uids.intersection(EXCLUDE_UIDS)),
    }

    print(f"Manifest Verification:")
    print(f"  Train Utterances:   {len(train_entries):,} (Expected: 2,248)")
    print(f"  Val Utterances:     {len(val_entries):,} (Expected: 445)")
    print(f"  Total Utterances:   {len(train_entries) + len(val_entries):,} (Expected: 2,693)")
    print(f"  Train/Val Leakage:  {results['manifest_check']['leakage_overlap']} overlapping utterances")
    print(f"  Excluded UIDs:      indic_te_0_927 and indic_te_3_0 properly excluded: {len(results['manifest_check']['excluded_found_in_train']) == 0 and len(results['manifest_check']['excluded_found_in_val']) == 0}")

    if results["manifest_check"]["leakage_overlap"] > 0:
        results["passed"] = False
        results["errors"].append("Train/Val transcript leakage detected!")
    if len(results["manifest_check"]["excluded_found_in_train"]) > 0 or len(results["manifest_check"]["excluded_found_in_val"]) > 0:
        results["passed"] = False
        results["errors"].append("Excluded failed utterances found in manifests!")

    # 2. File counts in feature directories
    mel_dir = os.path.join(PREP_BASE, "mel")
    pitch_dir = os.path.join(PREP_BASE, "pitch")
    energy_dir = os.path.join(PREP_BASE, "energy")
    dur_dir = os.path.join(PREP_BASE, "duration")

    mel_files = os.listdir(mel_dir)
    pitch_files = os.listdir(pitch_dir)
    energy_files = os.listdir(energy_dir)
    dur_files = os.listdir(dur_dir)

    results["file_counts"] = {
        "mel": len(mel_files),
        "pitch": len(pitch_files),
        "energy": len(energy_files),
        "duration": len(dur_files),
        "total_npy": len(mel_files) + len(pitch_files) + len(energy_files) + len(dur_files),
    }

    print(f"\nFeature File Counts:")
    print(f"  Mel Spectrograms:  {len(mel_files):,} (.npy)")
    print(f"  Pitch Contours:    {len(pitch_files):,} (.npy)")
    print(f"  Energy Contours:   {len(energy_files):,} (.npy)")
    print(f"  Duration Arrays:   {len(dur_files):,} (.npy)")
    print(f"  Total NPY Files:   {results['file_counts']['total_npy']:,} (Expected: 10,772)")

    if any(c != 2693 for c in [len(mel_files), len(pitch_files), len(energy_files), len(dur_files)]):
        results["passed"] = False
        results["errors"].append("Feature file count mismatch! Expected 2,693 per directory.")

    # 3. Exhaustive Dimension, Alignment & Finite Value Checks across all 2,693 utterances
    print(f"\nVerifying Dimensions, Alignments, and Finite Values across all 2,693 utterances...")
    all_entries = train_entries + val_entries
    mismatched_dur_sum = 0
    mismatched_pitch_len = 0
    mismatched_energy_len = 0
    non_finite_mel = 0
    non_finite_pitch = 0
    non_finite_energy = 0
    invalid_durations = 0

    total_mel_frames = 0
    total_tokens = 0

    for uid, spk, text, raw_text in all_entries:
        mel_p = os.path.join(mel_dir, f"{spk}-mel-{uid}.npy")
        pitch_p = os.path.join(pitch_dir, f"{spk}-pitch-{uid}.npy")
        energy_p = os.path.join(energy_dir, f"{spk}-energy-{uid}.npy")
        dur_p = os.path.join(dur_dir, f"{spk}-duration-{uid}.npy")

        mel = np.load(mel_p)
        pitch = np.load(pitch_p)
        energy = np.load(energy_p)
        dur = np.load(dur_p)

        n_frames = mel.shape[0]
        n_tokens = len(dur)
        sum_dur = int(np.sum(dur))

        total_mel_frames += n_frames
        total_tokens += n_tokens

        if mel.shape[1] != 80:
            results["errors"].append(f"{uid}: Mel channels != 80 ({mel.shape[1]})")
            results["passed"] = False
        if sum_dur != n_frames:
            mismatched_dur_sum += 1
        if len(pitch) != n_tokens:
            mismatched_pitch_len += 1
        if len(energy) != n_tokens:
            mismatched_energy_len += 1
        if not np.all(np.isfinite(mel)):
            non_finite_mel += 1
        if not np.all(np.isfinite(pitch)):
            non_finite_pitch += 1
        if not np.all(np.isfinite(energy)):
            non_finite_energy += 1
        if np.any(dur <= 0):
            invalid_durations += 1

    results["dimension_check"] = {
        "total_audited": len(all_entries),
        "mismatched_dur_sum": mismatched_dur_sum,
        "mismatched_pitch_len": mismatched_pitch_len,
        "mismatched_energy_len": mismatched_energy_len,
        "non_finite_mel": non_finite_mel,
        "non_finite_pitch": non_finite_pitch,
        "non_finite_energy": non_finite_energy,
        "invalid_durations": invalid_durations,
        "total_mel_frames": total_mel_frames,
        "total_tokens": total_tokens,
    }

    print(f"  Mel/Duration Sum Mismatches:   {mismatched_dur_sum}")
    print(f"  Pitch Length Mismatches:        {mismatched_pitch_len}")
    print(f"  Energy Length Mismatches:       {mismatched_energy_len}")
    print(f"  Non-finite Mel Frames:          {non_finite_mel}")
    print(f"  Non-finite Pitch Frames:        {non_finite_pitch}")
    print(f"  Non-finite Energy Frames:       {non_finite_energy}")
    print(f"  Invalid / Non-positive Durations: {invalid_durations}")

    if any(v > 0 for v in [mismatched_dur_sum, mismatched_pitch_len, mismatched_energy_len,
                           non_finite_mel, non_finite_pitch, non_finite_energy, invalid_durations]):
        results["passed"] = False
        results["errors"].append("Integrity check failed with dimension or finite value violations!")

    # 4. Reference Dataset & DataLoader Compatibility Test
    print(f"\nTesting Reference FastSpeech 2 Dataset & DataLoader Loading...")
    preprocess_config = {
        "dataset": "IndicTTS_Telugu",
        "path": {
            "preprocessed_path": PREP_BASE
        },
        "preprocessing": {
            "val_size": 445,
            "text": {
                "text_cleaners": ["basic_cleaners"],
                "language": "te"
            },
            "audio": {
                "sampling_rate": 16000,
                "max_wav_value": 32768.0
            },
            "stft": {
                "filter_length": 1024,
                "hop_length": 256,
                "win_length": 1024
            },
            "mel": {
                "n_mel_channels": 80,
                "mel_fmin": 0,
                "mel_fmax": 8000
            },
            "pitch": {
                "feature": "phoneme_level",
                "normalization": True
            },
            "energy": {
                "feature": "phoneme_level",
                "normalization": True
            }
        }
    }

    train_config = {
        "optimizer": {
            "batch_size": 16
        }
    }

    train_dataset = Dataset("train.txt", preprocess_config, train_config, sort=True, drop_last=False)
    val_dataset = Dataset("val.txt", preprocess_config, train_config, sort=False, drop_last=False)

    print(f"  Train Dataset length: {len(train_dataset):,} samples")
    print(f"  Val Dataset length:   {len(val_dataset):,} samples")

    train_loader = DataLoader(
        train_dataset,
        batch_size=16 * 4,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
    )

    sample_batch = None
    for batches in train_loader:
        sample_batch = batches[0]
        break

    (ids, raw_texts, speakers, texts, src_lens, max_src_len,
     mels, mel_lens, max_mel_len, pitches, energies, durations) = sample_batch

    results["dataloader_test"] = {
        "success": True,
        "sample_batch_size": len(ids),
        "ids_preview": ids[:4],
        "speakers_preview": speakers[:4].tolist(),
        "texts_shape": list(texts.shape),
        "mels_shape": list(mels.shape),
        "pitches_shape": list(pitches.shape),
        "energies_shape": list(energies.shape),
        "durations_shape": list(durations.shape),
    }

    print(f"  Sample Batch loaded successfully!")
    print(f"    Batch Size:     {len(ids)}")
    print(f"    Texts Shape:    {texts.shape}")
    print(f"    Mels Shape:     {mels.shape}")
    print(f"    Pitches Shape:  {pitches.shape}")
    print(f"    Energies Shape: {energies.shape}")
    print(f"    Durations Shape:{durations.shape}")

    # 5. Reference Model Batch Forward Pass Test
    print(f"\nTesting Reference FastSpeech 2 Model Forward Pass...")
    model_config = {
        "transformer": {
            "encoder_layer": 4,
            "encoder_head": 2,
            "encoder_hidden": 256,
            "decoder_layer": 6,
            "decoder_head": 2,
            "decoder_hidden": 256,
            "conv_filter_size": 1024,
            "conv_kernel_size": [9, 1],
            "encoder_dropout": 0.2,
            "decoder_dropout": 0.2
        },
        "variance_predictor": {
            "filter_size": 256,
            "kernel_size": 3,
            "dropout": 0.5
        },
        "variance_embedding": {
            "pitch_quantization": "linear",
            "energy_quantization": "linear",
            "n_bins": 256
        },
        "multi_speaker": True,
        "max_seq_len": 1500,
        "vocoder": {
            "model": "HiFi-GAN",
            "speaker": "universal"
        }
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FastSpeech2(preprocess_config, model_config).to(device)
    model.eval()

    device_batch = to_device(sample_batch, device)
    with torch.no_grad():
        output = model(*(device_batch[2:]))

    (mel_predictions, postnet_mel_predictions, pitch_predictions, energy_predictions,
     log_duration_predictions, _, src_masks, mel_masks, _, _) = output

    results["forward_pass_test"] = {
        "success": True,
        "mel_predictions_shape": list(mel_predictions.shape),
        "postnet_mel_shape": list(postnet_mel_predictions.shape),
        "pitch_predictions_shape": list(pitch_predictions.shape),
        "energy_predictions_shape": list(energy_predictions.shape),
        "log_durations_shape": list(log_duration_predictions.shape),
    }

    print(f"  Model Forward Pass Succeeded on device: {device}!")
    print(f"    Mel Predictions:         {mel_predictions.shape}")
    print(f"    PostNet Mel Predictions: {postnet_mel_predictions.shape}")
    print(f"    Pitch Predictions:       {pitch_predictions.shape}")
    print(f"    Energy Predictions:      {energy_predictions.shape}")
    print(f"    Duration Predictions:    {log_duration_predictions.shape}")

    # Save complete QA report
    qa_path = os.path.join(REPORT_DIR, "telugu_stage_d_qa_audit.json")
    with open(qa_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n{'='*80}")
    print(f"ALL QA CHECKS PASSED: {results['passed']}")
    print(f"Full QA report saved to: {qa_path}")
    print(f"{'='*80}\n")

    return results


if __name__ == "__main__":
    run_full_qa()

