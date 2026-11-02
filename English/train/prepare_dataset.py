"""
High-Speed Dataset Preparation & Integrity Verification for Kaggle FastSpeech 2.
Features:
- Idempotent: Skips re-copying if the dataset is already present and 100% verified.
- Fast Extraction & Copying: Uses optimized shell commands (unzip / cp) on Linux when available.
- Full Integrity Audit: Verifies train/val manifests, speaker map, stats, and feature tensors.
"""

import os
import sys
import glob
import json
import time
import shutil
import zipfile
import subprocess


def verify_dataset(target_dir):
    """
    Verifies that target_dir contains a complete, uncorrupted preprocessed dataset.
    Returns (True, details) if complete, (False, reason) otherwise.
    """
    preprocessed_dir = os.path.join(target_dir, "preprocessed_data")
    if not os.path.isdir(preprocessed_dir):
        return False, f"Missing preprocessed_data directory at {preprocessed_dir}"

    # Check manifest files
    train_txt = os.path.join(preprocessed_dir, "train.txt")
    val_txt = os.path.join(preprocessed_dir, "val.txt")
    stats_json = os.path.join(preprocessed_dir, "stats.json")
    speakers_json = os.path.join(preprocessed_dir, "speakers.json")

    for fpath, name in [(train_txt, "train.txt"), (val_txt, "val.txt"), 
                        (stats_json, "stats.json"), (speakers_json, "speakers.json")]:
        if not os.path.isfile(fpath) or os.path.getsize(fpath) == 0:
            return False, f"Missing or empty metadata file: {name}"

    try:
        with open(train_txt, "r", encoding="utf-8") as f:
            train_count = sum(1 for line in f if line.strip())
        with open(val_txt, "r", encoding="utf-8") as f:
            val_count = sum(1 for line in f if line.strip())
        with open(speakers_json, "r", encoding="utf-8") as f:
            speaker_map = json.load(f)
        with open(stats_json, "r", encoding="utf-8") as f:
            stats = json.load(f)
    except Exception as e:
        return False, f"Corrupted metadata JSON/text: {e}"

    if train_count != 3419:
        return False, f"train.txt line count mismatch: expected 3419, found {train_count}"
    if val_count != 746:
        return False, f"val.txt line count mismatch: expected 746, found {val_count}"
    if len(speaker_map) != 104:
        return False, f"speakers.json speaker count mismatch: expected 104, found {len(speaker_map)}"

    total_expected = train_count + val_count  # 4165 utterances

    # Verify feature subdirectories
    for feat in ["mel", "pitch", "energy", "duration"]:
        feat_dir = os.path.join(preprocessed_dir, feat)
        if not os.path.isdir(feat_dir):
            return False, f"Missing feature directory: {feat}"
        count = len(os.listdir(feat_dir))
        if count != total_expected:
            return False, f"Feature count mismatch for '{feat}': expected {total_expected}, found {count}"

    # Verify core codebase files
    for req_file in ["train.py", "config.yaml", "dataset.py", "evaluate.py", "inference.py"]:
        p = os.path.join(target_dir, req_file)
        if not os.path.isfile(p) or os.path.getsize(p) == 0:
            return False, f"Missing or empty core script: {req_file}"

    # Verify HiFi-GAN vocoder weights
    vocoder_file = os.path.join(target_dir, "hifigan", "generator_universal.pth.tar")
    if not os.path.isfile(vocoder_file) or os.path.getsize(vocoder_file) < 50_000_000:
        return False, f"Missing or incomplete HiFi-GAN vocoder weights at {vocoder_file}"

    details = {
        "train_utterances": train_count,
        "val_utterances": val_count,
        "total_utterances": total_expected,
        "total_feature_files": total_expected * 4,
        "unique_speakers": len(speaker_map),
        "vocoder_bytes": os.path.getsize(vocoder_file),
    }
    return True, details


def prepare_dataset(
    target_dir="/kaggle/working/English_FastSpeech2_Kaggle",
    search_dirs=None,
    force=False
):
    """
    Locates package/dataset from Kaggle input, copies/extracts into target_dir with speed optimizations,
    and runs full verification.
    """
    if search_dirs is None:
        search_dirs = [
            "/kaggle/input/datasets/saviocsb/english-fastspeech2-kaggle/English_FastSpeech2_Kaggle",
            "/kaggle/input/english-fastspeech2-kaggle/English_FastSpeech2_Kaggle",
            "/kaggle/input/english-fastspeech2-kaggle",
            "/kaggle/input",
        ]

    print("=" * 70)
    print("FAST DATASET PREPARATION & INTEGRITY VERIFICATION")
    print(f"Target Directory: {target_dir}")
    print("=" * 70)

    # Step 1: Idempotence check - is dataset already extracted and verified?
    if not force and os.path.exists(target_dir):
        print(f"[Check] Existing dataset found at {target_dir}. Verifying integrity...")
        is_valid, details = verify_dataset(target_dir)
        if is_valid:
            print("[SUCCESS] Existing dataset is complete, verified and ready for training!")
            print(f"  Utterances:     {details['train_utterances']:,} train, {details['val_utterances']:,} val")
            print(f"  Feature Files:  {details['total_feature_files']:,} (.npy)")
            print(f"  Unique Speakers:{details['unique_speakers']}")
            print(f"  Vocoder Size:   {details['vocoder_bytes'] / (1024**2):.2f} MB")
            print("[Info] Skipping unnecessary copying.")
            return target_dir
        else:
            print(f"[Notice] Existing dataset failed verification ({details}). Re-preparing...")

    # Step 2: Locate source in Kaggle input
    start_time = time.time()
    source_found = None
    is_zip = False

    # Check known paths first
    for path in search_dirs:
        if os.path.isdir(path) and os.path.exists(os.path.join(path, "preprocessed_data")):
            source_found = path
            is_zip = False
            break
        elif os.path.isfile(path) and path.endswith(".zip"):
            source_found = path
            is_zip = True
            break

    # If not found directly, glob for .zip or directory in /kaggle/input
    if source_found is None:
        print("[Search] Scanning /kaggle/input for package or dataset...")
        zip_candidates = glob.glob("/kaggle/input/**/English_FastSpeech2_Kaggle.zip", recursive=True)
        if not zip_candidates:
            zip_candidates = glob.glob("/kaggle/input/*/*.zip")
        
        if zip_candidates:
            source_found = zip_candidates[0]
            is_zip = True
        else:
            dir_candidates = glob.glob("/kaggle/input/**/preprocessed_data", recursive=True)
            if dir_candidates:
                source_found = os.path.dirname(dir_candidates[0])
                is_zip = False

    if source_found is None:
        raise FileNotFoundError(
            "Could not locate English FastSpeech 2 dataset or zip archive in /kaggle/input/! "
            "Please ensure the dataset is attached to this Kaggle notebook."
        )

    print(f"[Source Found] Type: {'ZIP Archive' if is_zip else 'Directory'}")
    print(f"  Location: {source_found}")

    os.makedirs(target_dir, exist_ok=True)

    # Step 3: Copy or extract using optimized methods
    if is_zip:
        print(f"[Extract] Extracting ZIP package into {os.path.dirname(target_dir)}...")
        # Check if unzip CLI is available for fast C-based extraction
        unzip_bin = shutil.which("unzip")
        parent_dir = os.path.dirname(target_dir)
        if unzip_bin:
            cmd = [unzip_bin, "-q", "-o", source_found, "-d", parent_dir]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                print(f"[Warning] Linux unzip failed ({res.stderr}). Falling back to python zipfile...")
                with zipfile.ZipFile(source_found, "r") as zf:
                    zf.extractall(parent_dir)
        else:
            with zipfile.ZipFile(source_found, "r") as zf:
                zf.extractall(parent_dir)

        # Handle nested subfolder if present
        if not os.path.exists(os.path.join(target_dir, "train.py")):
            nested = os.path.join(parent_dir, "English_FastSpeech2_Kaggle")
            if os.path.exists(nested) and nested != target_dir:
                for item in os.listdir(nested):
                    shutil.move(os.path.join(nested, item), target_dir)
    else:
        print(f"[Fast Setup] Setting up package from {source_found} to {target_dir}...")
        for item in os.listdir(source_found):
            s_item = os.path.join(source_found, item)
            t_item = os.path.join(target_dir, item)
            if item == "preprocessed_data":
                if os.path.islink(t_item) or os.path.exists(t_item):
                    if os.path.islink(t_item): os.unlink(t_item)
                    else: shutil.rmtree(t_item)
                try:
                    os.symlink(s_item, t_item)
                    print(f"  [Instant Symlink] {item} -> {s_item} (0 MB copied)")
                except Exception as e:
                    print(f"  [Symlink Fallback] Symlink failed ({e}), copying...")
                    shutil.copytree(s_item, t_item, dirs_exist_ok=True)
            elif os.path.isdir(s_item):
                shutil.copytree(s_item, t_item, dirs_exist_ok=True)
            else:
                shutil.copy2(s_item, t_item)

    elapsed = time.time() - start_time
    print(f"[Copy Complete] Operation finished in {elapsed:.2f} seconds.")

    # Step 4: Verification of copied dataset
    print("[Verification] Verifying copied dataset integrity...")
    is_valid, details = verify_dataset(target_dir)
    if not is_valid:
        raise RuntimeError(f"Dataset verification failed after copying: {details}")

    print("=" * 70)
    print("DATASET PREPARATION & INTEGRITY VERIFICATION PASSED")
    print(f"  Utterances:     {details['train_utterances']:,} train, {details['val_utterances']:,} val")
    print(f"  Feature Files:  {details['total_feature_files']:,} (.npy)")
    print(f"  Unique Speakers:{details['unique_speakers']}")
    print(f"  Vocoder Weight: {details['vocoder_bytes'] / (1024**2):.2f} MB")
    print("=" * 70 + "\n")
    return target_dir


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "/kaggle/working/English_FastSpeech2_Kaggle"
    prepare_dataset(target)

