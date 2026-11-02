"""
Multi-Speaker Telugu FastSpeech 2 Training Script for Kaggle and Local Environments.
Includes:
- Robust Memory Management (RAM & GPU monitoring, gc.collect(), torch.cuda.empty_cache() at boundaries)
- optimizer.zero_grad(set_to_none=True) for memory deallocation
- Strict Checkpoint Retention Policy (Top 3 Best Models + 1 Resumable Latest Checkpoint)
- Checkpoint Metadata Index (checkpoints_index.json)
- Pre-save Disk Space Monitoring and Low-Space Safeguards
- Atomic Checkpoint Saving with Verification before Deletion
- Device-Agnostic Execution (CUDA / Multi-GPU / CPU)
- HiFi-GAN Universal Vocoder Integration
- Final Checkpoint & Inference Artifact Export
"""

import os
import sys
import argparse
import glob
import re
import shutil
import json
import time
import datetime
import gc
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

try:
    import psutil
except ImportError:
    psutil = None

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

# Ensure package root is in sys.path
PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from utils.model import get_model, get_vocoder, get_param_num
from utils.tools import to_device, log, synth_one_sample
from model import FastSpeech2Loss
from dataset import Dataset
from evaluate import evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==============================================================================
# SECTION 1: MEMORY & DISK MONITORING UTILITIES
# ==============================================================================

def get_memory_info():
    """Returns a dictionary containing RAM and GPU memory usage statistics."""
    info = {
        "ram_used_gb": 0.0,
        "ram_total_gb": 0.0,
        "ram_pct": 0.0,
        "gpus": []
    }
    if psutil is not None:
        try:
            vm = psutil.virtual_memory()
            info["ram_used_gb"] = vm.used / (1024 ** 3)
            info["ram_total_gb"] = vm.total / (1024 ** 3)
            info["ram_pct"] = vm.percent
        except Exception:
            pass

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            try:
                alloc = torch.cuda.memory_allocated(i) / (1024 ** 3)
                res = torch.cuda.memory_reserved(i) / (1024 ** 3)
                max_alloc = torch.cuda.max_memory_allocated(i) / (1024 ** 3)
                info["gpus"].append({
                    "device": i,
                    "name": torch.cuda.get_device_name(i),
                    "alloc_gb": alloc,
                    "reserved_gb": res,
                    "max_alloc_gb": max_alloc,
                })
            except Exception:
                pass
    return info


def format_memory_summary(mem_info=None):
    """Formats RAM and GPU stats into a compact single-line log message."""
    if mem_info is None:
        mem_info = get_memory_info()
    parts = []
    if mem_info["ram_total_gb"] > 0:
        parts.append(f"RAM: {mem_info['ram_used_gb']:.2f}/{mem_info['ram_total_gb']:.2f} GB ({mem_info['ram_pct']:.1f}%)")
    for gpu in mem_info["gpus"]:
        parts.append(f"GPU[{gpu['device']}]: Alloc {gpu['alloc_gb']:.2f} GB, Res {gpu['reserved_gb']:.2f} GB")
    return " | ".join(parts) if parts else "Memory info unavailable"


def get_disk_free_gb(path):
    """Returns available disk space in gigabytes for the path."""
    try:
        check_path = path if os.path.exists(path) else os.path.dirname(os.path.abspath(path))
        if not os.path.exists(check_path):
            check_path = "."
        usage = shutil.disk_usage(check_path)
        return usage.free / (1024 ** 3)
    except Exception:
        return 999.0


# ==============================================================================
# SECTION 2: ATOMIC & VERIFIED CHECKPOINT SAVING
# ==============================================================================

def save_and_verify_checkpoint(payload, target_path, min_bytes=1000):
    """
    Atomically writes a checkpoint dictionary to target_path via a .tmp file.
    Verifies that the written file exists, meets minimum size, and loads cleanly.
    Returns True upon success, raises an exception if verification fails.
    """
    target_dir = os.path.dirname(os.path.abspath(target_path))
    os.makedirs(target_dir, exist_ok=True)
    temp_path = target_path + ".tmp"

    if os.path.exists(temp_path):
        try:
            os.remove(temp_path)
        except Exception:
            pass

    # Save to temporary path
    torch.save(payload, temp_path)

    # Verification checks
    if not os.path.isfile(temp_path):
        raise IOError(f"Checkpoint save verification failed: {temp_path} was not created.")

    sz = os.path.getsize(temp_path)
    if sz < min_bytes:
        os.remove(temp_path)
        raise ValueError(f"Checkpoint save verification failed: {temp_path} is too small ({sz} bytes).")

    # Integrity verification load (mmap=True to avoid heap allocations)
    try:
        try:
            test_load = torch.load(temp_path, map_location="cpu", weights_only=False, mmap=True)
        except (TypeError, ValueError):
            test_load = torch.load(temp_path, map_location="cpu", weights_only=False)
        if "model" not in test_load or "step" not in test_load:
            raise KeyError("Checkpoint missing required 'model' or 'step' key.")
        del test_load
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise IOError(f"Checkpoint integrity verification load failed: {e}")

    # Atomic swap
    os.replace(temp_path, target_path)
    gc.collect()
    if sys.platform.startswith("linux"):
        try:
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass
    return True


# ==============================================================================
# SECTION 3: STRICT CHECKPOINT RETENTION MANAGER
# ==============================================================================

class CheckpointRetentionManager:
    """
    Strict Checkpoint Retention Manager:
    - Retains exactly the top 3 best model checkpoints ranked by lowest validation loss.
    - Retains 1 separate latest resumable checkpoint containing full training state.
    - Maintains an index file: checkpoints_index.json.
    - Pre-verifies new checkpoints before deleting older ones.
    - Automatically prunes obsolete numbered checkpoints and temporary files.
    - Preemptively monitors disk space with emergency warnings and safeguards.
    """

    def __init__(self, ckpt_dir, max_best=3):
        self.ckpt_dir = os.path.abspath(ckpt_dir)
        self.max_best = max_best
        self.index_file = os.path.join(self.ckpt_dir, "checkpoints_index.json")
        self.best_checkpoints = []  # list of dicts: {"rank": int, "step": int, "val_loss": float, "filename": str}
        self.latest_info = None      # dict: {"step": int, "epoch": int, "val_loss": float, "best_loss": float, "filename": str}
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self._load_or_reconstruct_index()

    def _load_or_reconstruct_index(self):
        """Loads checkpoints_index.json if present; otherwise reconstructs it from existing files."""
        if os.path.isfile(self.index_file):
            try:
                with open(self.index_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.best_checkpoints = [
                    item for item in data.get("best_checkpoints", [])
                    if os.path.isfile(os.path.join(self.ckpt_dir, item.get("filename", "")))
                ]
                self.best_checkpoints.sort(key=lambda x: x["val_loss"])
                self.latest_info = data.get("latest_checkpoint", None)
                print(f"[RetentionManager] Loaded index with {len(self.best_checkpoints)} best checkpoint(s).")
                return
            except Exception as e:
                print(f"[RetentionManager] Warning: could not parse existing index ({e}). Reconstructing...")

        # Reconstruct from directory files
        self.best_checkpoints = []
        found_best = []
        for fpath in glob.glob(os.path.join(self.ckpt_dir, "best_step_*.pth")):
            fname = os.path.basename(fpath)
            m = re.search(r"best_step_(\d+)\.pth", fname)
            step_num = int(m.group(1)) if m else 0
            try:
                ckpt = torch.load(fpath, map_location="cpu", weights_only=False)
                val_loss = float(ckpt.get("val_loss", ckpt.get("best_loss", 999.0)))
                found_best.append({"step": step_num, "val_loss": val_loss, "filename": fname})
            except Exception:
                pass

        found_best.sort(key=lambda x: x["val_loss"])
        self.best_checkpoints = found_best[: self.max_best]
        for rank, item in enumerate(self.best_checkpoints, 1):
            item["rank"] = rank

        latest_path = os.path.join(self.ckpt_dir, "latest_checkpoint.pth")
        if os.path.isfile(latest_path):
            try:
                ckpt = torch.load(latest_path, map_location="cpu", weights_only=False)
                self.latest_info = {
                    "step": ckpt.get("step", 0),
                    "epoch": ckpt.get("epoch", 1),
                    "best_loss": ckpt.get("best_loss", 999.0),
                    "val_loss": ckpt.get("val_loss", 999.0),
                    "filename": "latest_checkpoint.pth",
                }
            except Exception:
                self.latest_info = None

        self._save_index()

    def _save_index(self):
        """Persists the checkpoint metadata index atomically to disk."""
        data = {
            "max_best": self.max_best,
            "best_checkpoints": self.best_checkpoints,
            "latest_checkpoint": self.latest_info,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        }
        idx_tmp = self.index_file + ".tmp"
        with open(idx_tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(idx_tmp, self.index_file)

    def check_disk_safeguard(self, min_gb=1.5, critical_gb=0.5):
        """Checks available disk space and raises or warns if running critically low."""
        free_gb = get_disk_free_gb(self.ckpt_dir)
        if free_gb < critical_gb:
            raise IOError(
                f"[CRITICAL DISK ERROR] Only {free_gb:.2f} GB free in {self.ckpt_dir}! "
                f"Halting checkpoint writes to prevent catastrophic filesystem corruption."
            )
        elif free_gb < min_gb:
            print(f"[DISK WARNING] Free disk space low: {free_gb:.2f} GB available. Triggering emergency cleanup...")
            self.prune_obsolete_checkpoints()

    def register_validation_result(self, step, epoch, val_loss, model, optimizer, best_loss):
        """
        Evaluates whether current validation loss enters top 3 best models.
        If yes, saves best_step_{step}.pth, updates ranking, evicts worst if > 5,
        and ensures best_model.pth points to the global #1 model.
        """
        self.check_disk_safeguard()

        qualifies = False
        if len(self.best_checkpoints) < self.max_best:
            qualifies = True
        elif val_loss < self.best_checkpoints[-1]["val_loss"]:
            qualifies = True

        raw_model = model.module if hasattr(model, "module") else model
        is_global_best = val_loss < best_loss

        if qualifies:
            filename = f"best_step_{step}.pth"
            target_path = os.path.join(self.ckpt_dir, filename)
            payload = {
                "model": raw_model.state_dict(),
                "optimizer": optimizer._optimizer.state_dict(),
                "step": step,
                "epoch": epoch,
                "val_loss": float(val_loss),
                "best_loss": float(min(best_loss, val_loss)),
                "saved_at": time.time(),
            }

            # Atomic save with verification
            save_and_verify_checkpoint(payload, target_path)

            # Update list
            # Remove any existing entry for this step
            self.best_checkpoints = [x for x in self.best_checkpoints if x["step"] != step]
            self.best_checkpoints.append({
                "step": step,
                "val_loss": float(val_loss),
                "filename": filename,
            })
            self.best_checkpoints.sort(key=lambda x: x["val_loss"])

            # Evict worst checkpoint if exceeding max_best
            while len(self.best_checkpoints) > self.max_best:
                evicted = self.best_checkpoints.pop()
                evicted_path = os.path.join(self.ckpt_dir, evicted["filename"])
                if os.path.isfile(evicted_path):
                    try:
                        os.remove(evicted_path)
                        print(f"[RetentionManager] Evicted 4th best checkpoint: {evicted['filename']} (Loss: {evicted['val_loss']:.4f})")
                    except Exception as e:
                        print(f"[RetentionManager] Warning removing evicted checkpoint {evicted_path}: {e}")

            # Re-assign ranks 1..max_best
            for rank, item in enumerate(self.best_checkpoints, 1):
                item["rank"] = rank

            # Always maintain best_model.pth matching rank 1
            if len(self.best_checkpoints) > 0:
                rank1_path = os.path.join(self.ckpt_dir, self.best_checkpoints[0]["filename"])
                best_model_path = os.path.join(self.ckpt_dir, "best_model.pth")
                try:
                    shutil.copyfile(rank1_path, best_model_path)
                except Exception as e:
                    print(f"[RetentionManager] Warning copying rank 1 to best_model.pth: {e}")

            self._save_index()
            print(f"[RetentionManager] Checkpoint qualified for TOP {self.max_best} -> {filename} (Val Loss: {val_loss:.4f})")

        return is_global_best

    def save_latest_resumable(self, step, epoch, val_loss, best_loss, model, optimizer):
        """
        Saves the latest full resumable training checkpoint to latest_checkpoint.pth.
        Always retains complete optimizer, scheduler, model, step, and epoch states.
        """
        self.check_disk_safeguard()

        raw_model = model.module if hasattr(model, "module") else model
        target_path = os.path.join(self.ckpt_dir, "latest_checkpoint.pth")
        payload = {
            "model": raw_model.state_dict(),
            "optimizer": optimizer._optimizer.state_dict(),
            "step": step,
            "epoch": epoch,
            "val_loss": float(val_loss) if val_loss is not None else None,
            "best_loss": float(best_loss) if best_loss is not None else None,
            "saved_at": time.time(),
            "rng_state": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            }
        }

        save_and_verify_checkpoint(payload, target_path)

        self.latest_info = {
            "step": step,
            "epoch": epoch,
            "val_loss": float(val_loss) if val_loss is not None else None,
            "best_loss": float(best_loss) if best_loss is not None else None,
            "filename": "latest_checkpoint.pth",
        }
        self._save_index()

        # Prune obsolete checkpoints automatically
        self.prune_obsolete_checkpoints()

    def prune_obsolete_checkpoints(self):
        """
        Scans ckpt_dir and removes:
        - Obsolete step_*.pth or *.pth.tar files not belonging to top 5 or latest.
        - Dangling .tmp files from interrupted writes.
        - Redundant duplicate copies.
        Guarantees that latest_checkpoint.pth and top 3 best models are never deleted.
        """
        latest_file = os.path.join(self.ckpt_dir, "latest_checkpoint.pth")
        if not os.path.isfile(latest_file):
            # Never prune if latest_checkpoint.pth doesn't exist
            return

        # Whitelist of filenames to protect
        protected = set()
        protected.add("latest_checkpoint.pth")
        protected.add("best_model.pth")
        protected.add("checkpoints_index.json")

        for item in self.best_checkpoints:
            protected.add(item["filename"])

        # Scan for candidate files to clean
        patterns = ["step_*.pth", "step_*.pth.tar", "*.pth.tar", "*.tmp", "best_step_*.pth"]
        pruned_count = 0
        freed_bytes = 0

        for pattern in patterns:
            for fpath in glob.glob(os.path.join(self.ckpt_dir, pattern)):
                fname = os.path.basename(fpath)
                if fname not in protected:
                    try:
                        sz = os.path.getsize(fpath)
                        os.remove(fpath)
                        pruned_count += 1
                        freed_bytes += sz
                    except Exception as e:
                        print(f"[RetentionManager] Warning deleting obsolete file {fname}: {e}")

        if pruned_count > 0:
            print(f"[RetentionManager] Pruned {pruned_count} obsolete checkpoint/temp file(s), freed {freed_bytes / (1024**2):.1f} MB.")


# ==============================================================================
# SECTION 4: ROBUST CHECKPOINT RESTORATION & RESUME
# ==============================================================================

def find_latest_valid_checkpoint(ckpt_dir):
    """
    Finds the latest valid checkpoint in ckpt_dir.
    Checks latest_checkpoint.pth first.
    Checks package checkpoints and fallback locations if ckpt_dir is initially empty.
    Falls back to checkpoints_index.json and historical .pth files in descending step order.
    Returns (ckpt_dict, ckpt_path, step) or (None, None, 0).
    """
    os.makedirs(ckpt_dir, exist_ok=True)

    # Check if ckpt_dir lacks latest_checkpoint.pth; search fallback locations
    latest_file = os.path.join(ckpt_dir, "latest_checkpoint.pth")
    if not os.path.isfile(latest_file):
        fallback_candidates = [
            os.path.join(PACKAGE_ROOT, "checkpoints"),
            os.path.join(PACKAGE_ROOT, "..", "checkpoints"),
            "/kaggle/working/checkpoints",
        ]
        # Also discover any checkpoints present in /kaggle/input
        for d in glob.glob("/kaggle/input/**/checkpoints", recursive=True):
            if d not in fallback_candidates:
                fallback_candidates.append(d)

        for fb in fallback_candidates:
            fb_latest = os.path.join(fb, "latest_checkpoint.pth")
            if os.path.isfile(fb_latest):
                print(f"[Checkpoint Auto-Discovery] Found pre-loaded checkpoints at {fb}.")
                print(f"  Copying to writable training directory: {ckpt_dir} ...")
                for fn in os.listdir(fb):
                    src_f = os.path.join(fb, fn)
                    dst_f = os.path.join(ckpt_dir, fn)
                    if not os.path.exists(dst_f):
                        try:
                            if os.path.isdir(src_f):
                                shutil.copytree(src_f, dst_f)
                            else:
                                shutil.copy2(src_f, dst_f)
                            print(f"  -> Imported: {fn}")
                        except Exception as e:
                            print(f"  [Warning] Failed to import {fn}: {e}")
                break

    if os.path.isfile(latest_file):
        try:
            ckpt = torch.load(latest_file, map_location=device, weights_only=False)
            step = ckpt.get("step", 0)
            print(f"[Checkpoint] Successfully loaded latest_checkpoint.pth (Step {step:,})")
            return ckpt, latest_file, step
        except Exception as e:
            print(f"[Warning] latest_checkpoint.pth appears corrupted: {e}. Falling back to historical checkpoints...")

    # Search indexed or historical candidates
    candidates = []
    search_dirs = [ckpt_dir, os.path.join(PACKAGE_ROOT, "checkpoints")]
    for d in search_dirs:
        if os.path.exists(d):
            for pattern in ["best_step_*.pth", "step_*.pth", "best_model.pth"]:
                for fpath in glob.glob(os.path.join(d, pattern)):
                    fname = os.path.basename(fpath)
                    m = re.search(r"(\d+)", fname)
                    step_num = int(m.group(1)) if m else 0
                    candidates.append((step_num, fpath))

    candidates.sort(key=lambda x: x[0], reverse=True)

    for step_num, fpath in candidates:
        try:
            ckpt = torch.load(fpath, map_location=device, weights_only=False)
            actual_step = ckpt.get("step", step_num)
            print(f"[Checkpoint] Successfully recovered from fallback checkpoint: {os.path.basename(fpath)} (Step {actual_step:,})")
            if os.path.dirname(os.path.abspath(fpath)) != os.path.abspath(ckpt_dir):
                shutil.copy2(fpath, os.path.join(ckpt_dir, "latest_checkpoint.pth"))
            return ckpt, fpath, actual_step
        except Exception as e:
            print(f"[Warning] Failed to load candidate {fpath}: {e}")

    print("[Checkpoint] No valid existing checkpoint found. Starting from scratch.")
    return None, None, 0


# ==============================================================================
# SECTION 5: FINAL EXPORT UTILITIES
# ==============================================================================

def export_final_weights(ckpt_dir, export_dir, vocoder_src=None, config_src=None, stats_src=None, speakers_src=None):
    """
    Exports final model weights, vocoder, and inference dependencies into a dedicated clean directory.
    Packages:
    - best_model.pth (Rank 1 best model)
    - latest_checkpoint.pth (Latest resumable recovery model)
    - Up to 5 best checkpoints
    - checkpoints_index.json
    - HiFi-GAN vocoder generator_universal.pth.tar
    - config.yaml, stats.json, speakers.json
    Verifies that every required file exists and has size > 0.
    Returns a dictionary of exported files and their byte sizes.
    """
    os.makedirs(export_dir, exist_ok=True)
    print("\n" + "=" * 70)
    print("FINAL MODEL CHECKPOINT & WEIGHT EXPORT")
    print(f"  Source Checkpoint Dir: {ckpt_dir}")
    print(f"  Target Export Dir:     {export_dir}")
    print("=" * 70)

    # 1. Best model
    best_model_src = os.path.join(ckpt_dir, "best_model.pth")
    if not os.path.isfile(best_model_src):
        # Look for best_step_*.pth
        b_files = glob.glob(os.path.join(ckpt_dir, "best_step_*.pth"))
        if b_files:
            b_files.sort()
            best_model_src = b_files[0]
        else:
            best_model_src = os.path.join(ckpt_dir, "latest_checkpoint.pth")

    if not os.path.isfile(best_model_src):
        raise FileNotFoundError(f"[Export Error] Required best_model.pth not found in {ckpt_dir}")
    shutil.copy2(best_model_src, os.path.join(export_dir, "best_model.pth"))

    # 2. Latest checkpoint
    latest_ckpt_src = os.path.join(ckpt_dir, "latest_checkpoint.pth")
    if not os.path.isfile(latest_ckpt_src):
        raise FileNotFoundError(f"[Export Error] Required latest_checkpoint.pth not found in {ckpt_dir}")
    shutil.copy2(latest_ckpt_src, os.path.join(export_dir, "latest_checkpoint.pth"))

    # 3. Best retained checkpoints
    for b_path in glob.glob(os.path.join(ckpt_dir, "best_step_*.pth")):
        shutil.copy2(b_path, os.path.join(export_dir, os.path.basename(b_path)))

    # 4. Checkpoint index
    idx_path = os.path.join(ckpt_dir, "checkpoints_index.json")
    if os.path.isfile(idx_path):
        shutil.copy2(idx_path, os.path.join(export_dir, "checkpoints_index.json"))

    # 5. HiFi-GAN Vocoder checkpoint
    if vocoder_src and os.path.isfile(vocoder_src):
        dst_vocoder = os.path.join(export_dir, os.path.basename(vocoder_src))
        shutil.copy2(vocoder_src, dst_vocoder)
    else:
        raise FileNotFoundError(f"[Export Error] Required HiFi-GAN vocoder file not found: {vocoder_src}")

    # 6. Core metadata for inference reconstruction
    if config_src and os.path.isfile(config_src):
        shutil.copy2(config_src, os.path.join(export_dir, "config.yaml"))
    if stats_src and os.path.isfile(stats_src):
        shutil.copy2(stats_src, os.path.join(export_dir, "stats.json"))
    if speakers_src and os.path.isfile(speakers_src):
        shutil.copy2(speakers_src, os.path.join(export_dir, "speakers.json"))

    # 7. Verification of exported files
    exported_summary = {}
    print("\n--- Exported Artifact Verification ---")
    print(f"{'Filename':<35} {'Size (Bytes)':<16} {'Size (MB)':<12} {'Status':<10}")
    print("-" * 75)
    for fname in sorted(os.listdir(export_dir)):
        fpath = os.path.join(export_dir, fname)
        if os.path.isfile(fpath):
            sz = os.path.getsize(fpath)
            if sz == 0:
                raise ValueError(f"[Export Error] Exported file {fname} is 0 bytes (corrupted)!")
            exported_summary[fname] = sz
            print(f"{fname:<35} {sz:<16,} {sz / (1024**2):<12.2f} {'VALID'}")

    print("-" * 75)
    print(f"[Export] Successfully verified {len(exported_summary)} files in {export_dir}.")
    print("=" * 70 + "\n")
    return exported_summary


# ==============================================================================
# SECTION 6: CONFIGURATION LOADER
# ==============================================================================


def update_best_model_folder(best_model_dir, ckpt_dir, vocoder_dir, config_src, stats_src, speakers_src, pkg_root):
    """
    Directly updates the Telugu_TTS_Best_Model folder with the latest best model weights
    and all required inference files whenever a new best model is obtained.
    Explicitly does NOT create or recreate any ZIP archive.
    """
    try:
        os.makedirs(best_model_dir, exist_ok=True)
        best_model_src = os.path.join(ckpt_dir, "best_model.pth")
        if not os.path.isfile(best_model_src):
            b_files = glob.glob(os.path.join(ckpt_dir, "best_step_*.pth"))
            if b_files:
                b_files.sort()
                best_model_src = b_files[0]
            else:
                best_model_src = os.path.join(ckpt_dir, "latest_checkpoint.pth")

        if os.path.isfile(best_model_src):
            shutil.copy2(best_model_src, os.path.join(best_model_dir, "best_model.pth"))
            os.makedirs(os.path.join(best_model_dir, "checkpoints"), exist_ok=True)
            shutil.copy2(best_model_src, os.path.join(best_model_dir, "checkpoints", "best_model.pth"))

        if config_src and os.path.isfile(config_src):
            shutil.copy2(config_src, os.path.join(best_model_dir, "config.yaml"))

        prep_dir = os.path.join(best_model_dir, "preprocessing")
        os.makedirs(prep_dir, exist_ok=True)
        if stats_src and os.path.isfile(stats_src):
            shutil.copy2(stats_src, os.path.join(best_model_dir, "stats.json"))
            shutil.copy2(stats_src, os.path.join(prep_dir, "stats.json"))
        if speakers_src and os.path.isfile(speakers_src):
            shutil.copy2(speakers_src, os.path.join(best_model_dir, "speakers.json"))
            shutil.copy2(speakers_src, os.path.join(prep_dir, "speakers.json"))

        # HiFi-GAN Vocoder
        if os.path.isdir(vocoder_dir):
            shutil.copytree(vocoder_dir, os.path.join(best_model_dir, "hifigan"), dirs_exist_ok=True)
        elif os.path.isfile(vocoder_dir):
            os.makedirs(os.path.join(best_model_dir, "hifigan"), exist_ok=True)
            shutil.copy2(vocoder_dir, os.path.join(best_model_dir, "hifigan", os.path.basename(vocoder_dir)))

        # Inference scripts
        for script_name in ["inference.py", "generate_assignment_audios.py", "dataset.py", "evaluate.py"]:
            sp = os.path.join(pkg_root, script_name)
            if os.path.isfile(sp):
                shutil.copy2(sp, os.path.join(best_model_dir, script_name))

        # Core code packages
        for pkg_name in ["model", "transformer", "text", "utils", "audio"]:
            src = os.path.join(pkg_root, pkg_name)
            dst = os.path.join(best_model_dir, pkg_name)
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)

        print(f"[BestModelSync] Directly updated '{best_model_dir}' with latest best model (NO zip archive created).")
    except Exception as e:
        print(f"[BestModelSync] Warning: Failed to update best model folder: {e}")


def load_configs(args):
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    preprocess_config = {
        "dataset": cfg["dataset"],
        "path": cfg["path"],
        "preprocessing": cfg["preprocessing"],
    }
    model_config = cfg["model"]
    train_config = cfg["train"]

    # CLI Overrides
    if args.data_dir:
        preprocess_config["path"]["preprocessed_path"] = os.path.abspath(args.data_dir)
    elif os.path.exists("/kaggle/working/Telugu_FastSpeech2_Kaggle/preprocessed_data"):
        preprocess_config["path"]["preprocessed_path"] = "/kaggle/working/Telugu_FastSpeech2_Kaggle/preprocessed_data"
    else:
        p_path = preprocess_config["path"]["preprocessed_path"]
        if not os.path.isabs(p_path):
            p_path = os.path.join(PACKAGE_ROOT, p_path)
        preprocess_config["path"]["preprocessed_path"] = p_path

    # Output directory handling (Kaggle friendly: strictly separated from Input)
    if args.output_dir:
        base_out = os.path.abspath(args.output_dir)
    elif os.path.exists("/kaggle/working"):
        base_out = "/kaggle/working/Telugu"
    else:
        base_out = os.path.join(PACKAGE_ROOT, "output")

    train_config["path"]["ckpt_path"] = os.path.join(base_out, "checkpoints")
    train_config["path"]["log_path"] = os.path.join(base_out, "logs")
    train_config["path"]["result_path"] = os.path.join(base_out, "results")

    # Step overrides if provided
    if args.total_step is not None:
        train_config["step"]["total_step"] = args.total_step
    if args.save_step is not None:
        train_config["step"]["save_step"] = args.save_step
    if args.val_step is not None:
        train_config["step"]["val_step"] = args.val_step
    if getattr(args, "synth_step", None) is not None:
        train_config["step"]["synth_step"] = args.synth_step
    if args.log_step is not None:
        train_config["step"]["log_step"] = args.log_step
    if args.batch_size is not None:
        train_config["optimizer"]["batch_size"] = args.batch_size

    return preprocess_config, model_config, train_config


# ==============================================================================
# SECTION 7: MAIN TRAINING LOOP WITH MEMORY & CHECKPOINT CONTROLS
# ==============================================================================

def train(args):
    configs = load_configs(args)
    preprocess_config, model_config, train_config = configs
    ckpt_dir = train_config["path"]["ckpt_path"]
    log_dir = train_config["path"]["log_path"]
    train_log_path = os.path.join(log_dir, "train")
    val_log_path = os.path.join(log_dir, "val")
    result_path = train_config["path"]["result_path"]

    export_dir = args.export_dir
    if not export_dir:
        export_dir = os.path.join(os.path.dirname(ckpt_dir.rstrip("/\\")), "Telugu_TTS_Best_Model")

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(train_log_path, exist_ok=True)
    os.makedirs(val_log_path, exist_ok=True)
    os.makedirs(result_path, exist_ok=True)

    print("=" * 70)
    print("Multi-Speaker Telugu FastSpeech 2 Training")
    print(f"Device:             {device}")
    print(f"Preprocessed Data:  {preprocess_config['path']['preprocessed_path']}")
    print(f"Checkpoint Output:  {ckpt_dir}")
    print(f"Log Output:         {log_dir}")
    print(f"Initial Disk Free:  {get_disk_free_gb(ckpt_dir):.2f} GB")
    print(f"Initial Memory:     {format_memory_summary()}")
    print("=" * 70)

    # Initialize Retention Manager
    retention_mgr = CheckpointRetentionManager(ckpt_dir=ckpt_dir, max_best=3)

    # Initialize Dataset and DataLoader
    print("[DataLoader] Initializing training dataset...")
    dataset = Dataset(
        "train.txt", preprocess_config, train_config, sort=True, drop_last=True
    )
    batch_size = train_config["optimizer"]["batch_size"]
    group_size = 4
    loader = DataLoader(
        dataset,
        batch_size=batch_size * group_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
        num_workers=0,
    )
    print(f"[DataLoader] Loaded {len(dataset):,} training utterances across {loader.batch_size} collated groups.")

    # Initialize Model and Optimizer
    print("[Model] Initializing FastSpeech 2 model and ScheduledOptim...")
    model, optimizer = get_model(args, configs, device, train=True)
    vocoder = get_vocoder(model_config, device)

    if torch.cuda.device_count() > 1:
        print(f"[Model] Using DataParallel across {torch.cuda.device_count()} GPUs.")
        model = nn.DataParallel(model)

    num_param = get_param_num(model)
    Loss = FastSpeech2Loss(preprocess_config, model_config).to(device)
    print(f"[Model] Total trainable parameters: {num_param:,}")

    # Loggers
    train_logger = SummaryWriter(train_log_path)
    val_logger = SummaryWriter(val_log_path)

    # Recovery / Checkpoint detection
    step = 1
    epoch = 1
    best_loss = float("inf")
    current_val_loss = None

    if args.resume and args.restore_step == 0:
        recovered_ckpt, ckpt_file, recovered_step = find_latest_valid_checkpoint(ckpt_dir)
        if recovered_ckpt is not None:
            raw_model = model.module if hasattr(model, "module") else model
            raw_model.load_state_dict(recovered_ckpt["model"])
            if "optimizer" in recovered_ckpt:
                optimizer.load_state_dict(recovered_ckpt["optimizer"])
            step = recovered_step + 1
            epoch = recovered_ckpt.get("epoch", 1)
            best_loss = recovered_ckpt.get("best_loss", float("inf"))
            current_val_loss = recovered_ckpt.get("val_loss", None)
            optimizer.current_step = step
            if "rng_state" in recovered_ckpt and recovered_ckpt["rng_state"]:
                rng = recovered_ckpt["rng_state"]
                if "torch" in rng and rng["torch"] is not None:
                    try:
                        t_state = rng["torch"]
                        if isinstance(t_state, torch.Tensor):
                            t_state = t_state.to(torch.uint8).cpu()
                        torch.set_rng_state(t_state)
                    except Exception as e:
                        print(f"[Warning] Skipped torch RNG restoration: {e}")
                if "cuda" in rng and rng["cuda"] is not None and torch.cuda.is_available():
                    try:
                        torch.cuda.set_rng_state_all(rng["cuda"])
                    except Exception:
                        pass
            print(f"[Resume] Resumed training state from {os.path.basename(ckpt_file)} at step {step:,}, epoch {epoch}, best_loss {best_loss:.4f}")
    elif args.restore_step > 0:
        step = args.restore_step + 1
        print(f"[Resume] Resumed from explicitly requested restore_step {args.restore_step}")

    total_step = train_config["step"]["total_step"]
    log_step = train_config["step"]["log_step"]
    save_step = train_config["step"]["save_step"]
    synth_step = train_config["step"]["synth_step"]
    val_step = train_config["step"]["val_step"]
    grad_acc_step = train_config["optimizer"]["grad_acc_step"]
    grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]
    print("\n" + "=" * 75)
    print(f"STARTING TRAINING LOOP: Target {total_step:,} Steps | Logging every {log_step} steps")
    print(f"Validation every {val_step} steps | Checkpoint saving every {save_step} steps")
    print("=" * 75 + "\n", flush=True)

    last_log_time = time.time()
    last_log_step = step
    training_start_time = time.time()

    try:
        while step <= total_step:
            for batchs in loader:
                for batch in batchs:
                    batch = to_device(batch, device)

                    # Forward pass
                    output = model(*(batch[2:]))

                    # Loss calculation
                    losses = Loss(batch, output)
                    total_loss = losses[0]

                    # Backward pass
                    total_loss = total_loss / grad_acc_step
                    total_loss.backward()

                    if step % grad_acc_step == 0:
                        nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)
                        optimizer.step_and_update_lr()
                        try:
                            optimizer.zero_grad(set_to_none=True)
                        except TypeError:
                            optimizer.zero_grad()

                    # Periodic step logging (strictly at regular intervals)
                    if step % log_step == 0:
                        now = time.time()
                        elapsed_interval = now - last_log_time
                        steps_in_interval = max(1, step - last_log_step)
                        ms_per_step = (elapsed_interval / steps_in_interval) * 1000.0

                        # Calculate ETA
                        steps_remaining = total_step - step
                        eta_seconds = int(steps_remaining * (elapsed_interval / steps_in_interval))
                        eta_str = str(datetime.timedelta(seconds=eta_seconds))

                        current_lr = optimizer._optimizer.param_groups[0]["lr"]
                        loss_vals = [l.item() for l in losses]

                        msg = (
                            f"[Step {step:6d}/{total_step:,} | Epoch {epoch:2d}] "
                            f"Loss: {loss_vals[0]:.4f} (Mel: {loss_vals[1]:.4f}, PostNet: {loss_vals[2]:.4f}, "
                            f"P: {loss_vals[3]:.4f}, E: {loss_vals[4]:.4f}, D: {loss_vals[5]:.4f}) | "
                            f"LR: {current_lr:.2e} | {ms_per_step:.1f}ms/step | ETA: {eta_str}"
                        )
                        print(msg, flush=True)
                        with open(os.path.join(train_log_path, "log.txt"), "a") as f:
                            f.write(msg + "\n")
                        log(train_logger, step, losses=loss_vals)

                        last_log_time = now
                        last_log_step = step
                        try:
                            train_logger.flush()
                        except Exception:
                            pass
                        # Host RAM leak prevention
                        if step % (log_step * 2) == 0:
                            gc.collect()
                            if sys.platform.startswith("linux"):
                                try:
                                    import ctypes
                                    ctypes.CDLL("libc.so.6").malloc_trim(0)
                                except Exception:
                                    pass

                    # Sample audio/spectrogram synthesis
                    if step % synth_step == 0:
                        try:
                            fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
                                batch,
                                output,
                                vocoder,
                                model_config,
                                preprocess_config,
                            )
                            log(train_logger, fig=fig, tag=f"Training/step_{step}_{tag}")
                            sr = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
                            log(train_logger, audio=wav_reconstruction, sampling_rate=sr, tag=f"Training/step_{step}_{tag}_reconstructed")
                            log(train_logger, audio=wav_prediction, sampling_rate=sr, tag=f"Training/step_{step}_{tag}_synthesized")
                            if plt is not None:
                                plt.close(fig)
                                plt.close("all")
                            print(f"  -> [Synthesis @ Step {step:,}] Generated audio and spectrogram validation sample.", flush=True)
                        except Exception as e:
                            print(f"  -> [Synthesis Warning @ Step {step}]: {e}", flush=True)

                    # Validation
                    if step % val_step == 0:
                        model.eval()
                        val_msg, val_losses = evaluate(model, step, configs, val_logger, vocoder)
                        with open(os.path.join(val_log_path, "log.txt"), "a") as f:
                            f.write(val_msg + "\n")

                        current_val_loss = val_losses[0]
                        print("\n" + "-" * 75, flush=True)
                        print(f"[Validation @ Step {step:,} | Epoch {epoch}] Val Loss: {current_val_loss:.4f} "
                              f"(Mel: {val_losses[1]:.4f}, PostNet: {val_losses[2]:.4f}, P: {val_losses[3]:.4f}, E: {val_losses[4]:.4f}, D: {val_losses[5]:.4f})", flush=True)

                        is_new_global_best = retention_mgr.register_validation_result(
                            step=step,
                            epoch=epoch,
                            val_loss=current_val_loss,
                            model=model,
                            optimizer=optimizer,
                            best_loss=best_loss,
                        )
                        if is_new_global_best:
                            best_loss = current_val_loss
                            print(f"*** New Global Best Model! Validation Loss: {best_loss:.4f} at Step {step:,} ***", flush=True)
                            if export_dir:
                                update_best_model_folder(
                                    best_model_dir=export_dir,
                                    ckpt_dir=ckpt_dir,
                                    vocoder_dir=os.path.join(PACKAGE_ROOT, "hifigan"),
                                    config_src=args.config if os.path.isfile(args.config) else os.path.join(PACKAGE_ROOT, "config.yaml"),
                                    stats_src=os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json"),
                                    speakers_src=os.path.join(preprocess_config["path"]["preprocessed_path"], "speakers.json"),
                                    pkg_root=PACKAGE_ROOT,
                                )

                        top_ckpts = [f"Rank {c.get('rank', i+1)}: Step {c['step']:,} ({c['val_loss']:.4f})" for i, c in enumerate(retention_mgr.best_checkpoints)]
                        print(f"[RetentionManager] Current Top {len(top_ckpts)} Models: {' | '.join(top_ckpts)}", flush=True)
                        print("-" * 75 + "\n", flush=True)

                        model.train()
                        try:
                            val_logger.flush()
                            train_logger.flush()
                        except Exception:
                            pass
                        if plt is not None:
                            plt.close("all")
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        if sys.platform.startswith("linux"):
                            try:
                                import ctypes
                                ctypes.CDLL("libc.so.6").malloc_trim(0)
                            except Exception:
                                pass

                    # Periodic Checkpoint Saving
                    if step % save_step == 0:
                        retention_mgr.save_latest_resumable(
                            step=step,
                            epoch=epoch,
                            val_loss=current_val_loss,
                            best_loss=best_loss,
                            model=model,
                            optimizer=optimizer,
                        )
                        print(f"  -> [Checkpoint @ Step {step:,}] Saved latest resumable checkpoint (latest_checkpoint.pth) | Free Disk: {get_disk_free_gb(ckpt_dir):.2f} GB", flush=True)

                    # Release iteration references
                    del output, losses, total_loss, batch

                    if step >= total_step:
                        total_elapsed = str(datetime.timedelta(seconds=int(time.time() - training_start_time)))
                        print("\n" + "=" * 75, flush=True)
                        print(f"Training completed successfully at step {step:,}! Total Elapsed: {total_elapsed}", flush=True)
                        print("=" * 75 + "\n", flush=True)
                        if export_dir:
                            update_best_model_folder(
                                best_model_dir=export_dir,
                                ckpt_dir=ckpt_dir,
                                vocoder_dir=os.path.join(PACKAGE_ROOT, "hifigan"),
                                config_src=args.config if os.path.isfile(args.config) else os.path.join(PACKAGE_ROOT, "config.yaml"),
                                stats_src=os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json"),
                                speakers_src=os.path.join(preprocess_config["path"]["preprocessed_path"], "speakers.json"),
                                pkg_root=PACKAGE_ROOT,
                            )
                        return

                    step += 1

            # Epoch boundary memory cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            epoch += 1

    except KeyboardInterrupt:
        print("\n[Interrupt] Training interrupted by user or environment. Saving latest emergency checkpoint...")
        retention_mgr.save_latest_resumable(
            step=step,
            epoch=epoch,
            val_loss=current_val_loss,
            best_loss=best_loss,
            model=model,
            optimizer=optimizer,
        )
        print(f"[Interrupt] Successfully preserved resumable state at step {step:,}. Exiting.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Speaker Telugu FastSpeech 2 Training")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to preprocessed data directory")
    parser.add_argument("--output_dir", type=str, default=None, help="Base output directory for checkpoints/logs")
    parser.add_argument("--export_dir", type=str, default=None, help="Target export directory for final model package")
    parser.add_argument("--restore_step", type=int, default=0, help="Explicit step number to restore")
    parser.add_argument("--resume", action="store_true", default=True, help="Auto-resume from latest checkpoint if available (default: True)")
    parser.add_argument("--no_resume", "--restart", action="store_false", dest="resume", help="Restart training from step 0 (ignore existing checkpoints)")
    parser.add_argument("--total_step", type=int, default=None, help="Override total steps")
    parser.add_argument("--save_step", type=int, default=None, help="Override save step interval")
    parser.add_argument("--val_step", type=int, default=None, help="Override validation interval")
    parser.add_argument("--synth_step", type=int, default=None, help="Override sample synthesis interval")
    parser.add_argument("--log_step", type=int, default=None, help="Override logging interval")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")

    args = parser.parse_args()
    train(args)
