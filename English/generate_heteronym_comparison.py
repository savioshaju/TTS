import os
import sys
import json
import yaml
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import wavfile

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = r"D:\Savio Shaju\TTS"
PKG_DIR = os.path.join(BASE_DIR, "kaggle_staging", "English_FastSpeech2_Kaggle")
if PKG_DIR not in sys.path:
    sys.path.insert(0, PKG_DIR)

from model import FastSpeech2
import hifigan
from utils.model import vocoder_infer
from text import text_to_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load configuration
config_path = os.path.join(BASE_DIR, "English", "config.yaml")
with open(config_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

cfg["path"]["preprocessed_path"] = os.path.join(BASE_DIR, "preprocessing", "English")
preprocess_config = {"dataset": cfg["dataset"], "path": cfg["path"], "preprocessing": cfg["preprocessing"]}
model_config = cfg["model"]

# Stats
stats_path = os.path.join(cfg["path"]["preprocessed_path"], "stats.json")
with open(stats_path, "r") as f:
    stats = json.load(f)
    pitch_mean, pitch_std = stats["pitch"][2], stats["pitch"][3]
    energy_mean, energy_std = stats["energy"][2], stats["energy"][3]

RESULTS_DIR = os.path.join(BASE_DIR, "Results", "English")
AUDIO_DIR = os.path.join(RESULTS_DIR, "test_generated_audios")
PLOT_DIR = os.path.join(RESULTS_DIR, "variance_adaptor_plots")

# Model & Vocoder
model = FastSpeech2(preprocess_config, model_config).to(device)
ckpt_path = os.path.join(BASE_DIR, "English", "checkpoints", "best_model.pth")
ckpt = torch.load(ckpt_path, map_location=device)
model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
model.eval()

hifigan_dir = os.path.join(PKG_DIR, "hifigan")
with open(os.path.join(hifigan_dir, "config.json"), "r") as f:
    v_cfg = hifigan.AttrDict(json.load(f))
vocoder = hifigan.Generator(v_cfg)
ckpt_voc = torch.load(os.path.join(hifigan_dir, "generator_universal.pth.tar"), map_location=device)
vocoder.load_state_dict(ckpt_voc["generator"])
vocoder.eval()
vocoder.remove_weight_norm()
vocoder.to(device)

target_spk_id = 65 # spk_067
cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
hop_length = preprocess_config["preprocessing"]["stft"]["hop_length"]

# Extract and Plot Heteronym Comparison: E2_a conduct vs E2_b conduct
# Load synthesis results summary
with open(os.path.join(RESULTS_DIR, "synthesis_analysis_summary.json"), "r", encoding="utf-8") as f:
    res = json.load(f)

# E2_a: tokens for conduct are at index:
tokens_a = res["E2_a"]["tokens"]
tokens_b = res["E2_b"]["tokens"]

idx_a = [i for i, t in enumerate(tokens_a) if t in ["k", "ɒ", "n", "ɖ", "ə", "ʈ"]][1:8] # 'k ɒ n ɖ ə k ʈ'
idx_b = [i for i, t in enumerate(tokens_b) if t in ["k", "ə", "n", "ɖ", "ʈ"]][:7] # 'k ə n ɖ ə k ʈ'

# Conduct noun tokens & metrics
cond_a_tokens = ["k", "ɒ", "n", "ɖ", "ə", "k", "ʈ"]
cond_a_dur = [res["E2_a"]["duration_ms"][tokens_a.index("k ɒ n ɖ ə k ʈ".split()[j])] if "k ɒ n ɖ ə k ʈ".split()[j] in tokens_a else 80.0 for j in range(7)]
# More accurately, find slice in tokens_a:
for i in range(len(tokens_a)-6):
    if tokens_a[i:i+7] == ["k", "ɒ", "n", "ɖ", "ə", "k", "ʈ"]:
        slice_a = range(i, i+7)
        break

for i in range(len(tokens_b)-6):
    if tokens_b[i:i+7] == ["k", "ə", "n", "ɖ", "ə", "k", "ʈ"]:
        slice_b = range(i, i+7)
        break

print(f"Slice A (conduct noun): {list(slice_a)} -> {[tokens_a[k] for k in slice_a]}")
print(f"Slice B (conduct verb): {list(slice_b)} -> {[tokens_b[k] for k in slice_b]}")

fig, axes = plt.subplots(3, 2, figsize=(13, 8), sharey="row")

# Row 1: Duration
dur_a = [res["E2_a"]["duration_ms"][k] for k in slice_a]
dur_b = [res["E2_b"]["duration_ms"][k] for k in slice_b]
tok_a = [tokens_a[k] for k in slice_a]
tok_b = [tokens_b[k] for k in slice_b]

axes[0, 0].bar(tok_a, dur_a, color="#1f77b4", alpha=0.85)
axes[0, 0].set_title("E2_a: Noun 'conduct' (/k ɒ n ɖ ə k ʈ/)\nTotal Duration: {:.1f} ms".format(sum(dur_a)), fontweight="bold")
axes[0, 0].set_ylabel("Duration (ms)", fontweight="bold")
axes[0, 0].grid(axis="y", linestyle="--", alpha=0.5)

axes[0, 1].bar(tok_b, dur_b, color="#ff7f0e", alpha=0.85)
axes[0, 1].set_title("E2_b: Verb 'conduct' (/k ə n ɖ ə k ʈ/)\nTotal Duration: {:.1f} ms".format(sum(dur_b)), fontweight="bold")
axes[0, 1].grid(axis="y", linestyle="--", alpha=0.5)

# Row 2: Pitch F0
pitch_a = [res["E2_a"]["pitch_hz"][k] for k in slice_a]
pitch_b = [res["E2_b"]["pitch_hz"][k] for k in slice_b]

axes[1, 0].plot(tok_a, pitch_a, marker="o", color="#d62728", linewidth=2.0)
axes[1, 0].set_title("Noun Pitch Contour (Trochaic Stress)", fontweight="bold")
axes[1, 0].set_ylabel("Pitch F0 (Hz)", fontweight="bold")
axes[1, 0].grid(True, linestyle="--", alpha=0.5)

axes[1, 1].plot(tok_b, pitch_b, marker="o", color="#d62728", linewidth=2.0)
axes[1, 1].set_title("Verb Pitch Contour (Iambic Stress)", fontweight="bold")
axes[1, 1].grid(True, linestyle="--", alpha=0.5)

# Row 3: Energy
energy_a = [res["E2_a"]["energy_db"][k] for k in slice_a]
energy_b = [res["E2_b"]["energy_db"][k] for k in slice_b]

axes[2, 0].plot(tok_a, energy_a, marker="s", color="#2ca02c", linewidth=2.0)
axes[2, 0].set_title("Noun Energy Contour (High initial vowel /ɒ/)", fontweight="bold")
axes[2, 0].set_ylabel("Energy (dB)", fontweight="bold")
axes[2, 0].set_xlabel("Phoneme", fontweight="bold")
axes[2, 0].grid(True, linestyle="--", alpha=0.5)

axes[2, 1].plot(tok_b, energy_b, marker="s", color="#2ca02c", linewidth=2.0)
axes[2, 1].set_title("Verb Energy Contour (Reduced vowel /ə/)", fontweight="bold")
axes[2, 1].set_xlabel("Phoneme", fontweight="bold")
axes[2, 1].grid(True, linestyle="--", alpha=0.5)

plt.suptitle("Heteronym Acoustic Comparison: FastSpeech 2 Prediction on 'conduct'", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
comp_plot_path = os.path.join(PLOT_DIR, "conduct_heteronym_comparison.png")
plt.savefig(comp_plot_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved heteronym comparison plot to: {comp_plot_path}")
