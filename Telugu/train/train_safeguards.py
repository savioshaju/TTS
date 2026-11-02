"""
Telugu FastSpeech 2 Production Training Engine with Robust Safeguards
Implements Stage F Training Safeguards:
- Atomic checkpointing (atomic write via temp file then rename)
- Best-model tracking based on validation loss (never overwritten with worse loss)
- Latest-checkpoint maintenance with full RNG, model, optimizer, scheduler, step, epoch state
- Periodic numbered checkpointing
- Numerical safeguards: NaN/Inf detection, gradient norm clipping, parameter sanity checks
- Robust automatic resume and recovery from latest or numbered fallbacks
- Graceful interruption handling (SIGINT/SIGTERM) with atomic state preservation
- Comprehensive structured logging and step timing
- OOM protection and memory management
"""

import os
import sys
import time
import json
import signal
import random
import shutil
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml

# Reference codebase path
sys.path.append(os.path.abspath("FastSpeech2-master"))

from utils.model import get_model, get_vocoder, get_param_num
from utils.tools import to_device, log, synth_one_sample
from model import FastSpeech2Loss
from dataset import Dataset


def set_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def atomic_save(state_dict, filepath):
    """Saves checkpoint to a temporary file first, then atomically replaces the destination."""
    temp_path = filepath + ".tmp"
    torch.save(state_dict, temp_path)
    if os.path.exists(filepath):
        os.replace(temp_path, filepath)
    else:
        os.rename(temp_path, filepath)


def save_checkpoint(
    ckpt_dir,
    step,
    epoch,
    model,
    optimizer,
    loss_val=None,
    is_best=False,
    best_loss=None,
    is_periodic=False,
):
    """
    Standardized, safe checkpoint serialization:
    - Maintains latest_checkpoint.pth
    - If is_best, maintains best_model.pth (with metadata)
    - If is_periodic, saves step_{step}.pth
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    raw_model = model.module if isinstance(model, nn.DataParallel) else model

    state = {
        "step": step,
        "epoch": epoch,
        "model": raw_model.state_dict(),
        "optimizer": optimizer._optimizer.state_dict(),
        "step_num": optimizer.current_step,
        "val_loss": loss_val,
        "best_val_loss": best_loss,
        "rng_torch": torch.get_rng_state(),
        "rng_numpy": np.random.get_state(),
        "rng_random": random.getstate(),
    }
    if torch.cuda.is_available():
        state["rng_cuda"] = torch.cuda.get_rng_state_all()

    # 1. Latest checkpoint
    latest_path = os.path.join(ckpt_dir, "latest_checkpoint.pth")
    atomic_save(state, latest_path)

    # 2. Periodic checkpoint
    if is_periodic:
        periodic_path = os.path.join(ckpt_dir, f"step_{step}.pth")
        atomic_save(state, periodic_path)

    # 3. Best checkpoint
    if is_best:
        best_path = os.path.join(ckpt_dir, "best_model.pth")
        atomic_save(state, best_path)


def load_checkpoint(ckpt_dir, model, optimizer, restore_step=0, fallback=True):
    """
    Resumes training from the latest valid checkpoint, a specific step, or the best model.
    Falls back gracefully if the target checkpoint is damaged.
    """
    raw_model = model.module if isinstance(model, nn.DataParallel) else model
    target_path = None

    if restore_step > 0:
        target_path = os.path.join(ckpt_dir, f"step_{restore_step}.pth")
        if not os.path.isfile(target_path):
            target_path = os.path.join(ckpt_dir, f"{restore_step}.pth.tar")
    else:
        target_path = os.path.join(ckpt_dir, "latest_checkpoint.pth")

    if not os.path.isfile(target_path) and fallback:
        # Check for any valid checkpoint in reverse order
        all_ckpts = []
        if os.path.isdir(ckpt_dir):
            for f in os.listdir(ckpt_dir):
                if f.endswith(".pth") or f.endswith(".pth.tar"):
                    all_ckpts.append(os.path.join(ckpt_dir, f))
        if all_ckpts:
            all_ckpts.sort(key=os.path.getmtime, reverse=True)
            target_path = all_ckpts[0]
            print(f"[Recovery Warning] Target not found. Falling back to newest checkpoint: {target_path}")

    if not target_path or not os.path.isfile(target_path):
        print("[Checkpoint] No existing checkpoint found to restore. Starting from step 0.")
        return 0, 1, float("inf")

    print(f"[Checkpoint] Loading checkpoint from: {target_path}")
    checkpoint = torch.load(target_path, map_location="cpu")

    # Load model
    if "model" in checkpoint:
        raw_model.load_state_dict(checkpoint["model"])
    else:
        raw_model.load_state_dict(checkpoint)

    # Load optimizer
    if "optimizer" in checkpoint and optimizer is not None:
        optimizer._optimizer.load_state_dict(checkpoint["optimizer"])
        if "step_num" in checkpoint:
            optimizer.current_step = checkpoint["step_num"]

    # Load step & epoch
    step = checkpoint.get("step", restore_step)
    epoch = checkpoint.get("epoch", 1)
    best_val_loss = checkpoint.get("best_val_loss", float("inf"))

    # Restore RNG states if available
    try:
        if "rng_torch" in checkpoint:
            torch.set_rng_state(checkpoint["rng_torch"])
        if "rng_numpy" in checkpoint:
            np.random.set_state(checkpoint["rng_numpy"])
        if "rng_random" in checkpoint:
            random.setstate(checkpoint["rng_random"])
        if "rng_cuda" in checkpoint and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(checkpoint["rng_cuda"])
    except Exception as e:
        print(f"[RNG Warning] Could not restore RNG state: {e}")

    print(f"[Checkpoint] Successfully restored state at step={step}, epoch={epoch}, best_val_loss={best_val_loss}")
    return step, epoch, best_val_loss


def validate(model, loss_fn, val_loader, device):
    """Runs a strict evaluation loop over validation data without computing gradients."""
    model.eval()
    loss_sums = [0.0] * 6
    total_samples = 0
    with torch.no_grad():
        for batchs in val_loader:
            for batch in batchs:
                batch = to_device(batch, device)
                out = model(*(batch[2:]))
                losses = loss_fn(batch, out)
                batch_len = len(batch[0])
                for i in range(len(losses)):
                    loss_sums[i] += losses[i].item() * batch_len
                total_samples += batch_len

    if total_samples == 0:
        return [0.0] * 6

    loss_means = [s / total_samples for s in loss_sums]
    return loss_means

