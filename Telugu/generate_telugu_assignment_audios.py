"""
Telugu FastSpeech 2 — Assignment Test Sentence Synthesis
=========================================================
Generates T1, T2_a, T2_b, T3, T4 audio files and variance
adaptor plots as required by the assignment benchmarks.

Checkpoints:
  - Model: Telugu FastSpeech 2 Phase 2 Best Model (Step 74,000, Val Loss = 2.1457)
  - Vocoder: HiFi-GAN Universal with 22,050 Hz -> 16,000 Hz resampling bridge
  - Speaker: telugufemale (Speaker ID: 0)
"""

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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
PKG_DIR = os.path.join(SCRIPT_DIR, "inference")
if not os.path.exists(PKG_DIR):
    PKG_DIR = os.path.join(BASE_DIR, "kaggle_staging", "Telugu_FastSpeech2_Kaggle")
if PKG_DIR not in sys.path:
    sys.path.insert(0, PKG_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from model import FastSpeech2
import hifigan
from utils.model import vocoder_infer, get_vocoder
from utils.tools import get_mask_from_lengths
from text import text_to_sequence
import unicodedata
import re
import argparse

TELUGU_VOWEL_TOKENS = {
    '@a', '@aa', '@i', '@ii', '@u', '@uu', '@ru', '@e', '@ee', '@ai', '@o', '@oo', '@au'
}

TELUGU_VOWELS = {
    '\u0c05': 'a', '\u0c06': 'aa', '\u0c07': 'i', '\u0c08': 'ii',
    '\u0c09': 'u', '\u0c0a': 'uu', '\u0c0b': 'ru', '\u0c0e': 'e',
    '\u0c0f': 'ee', '\u0c10': 'ai', '\u0c12': 'o', '\u0c13': 'oo', '\u0c14': 'au'
}
TELUGU_MATRAS = {
    '\u0c3e': 'aa', '\u0c3f': 'i', '\u0c40': 'ii', '\u0c41': 'u',
    '\u0c42': 'uu', '\u0c43': 'ru', '\u0c46': 'e', '\u0c47': 'ee',
    '\u0c48': 'ai', '\u0c4a': 'o', '\u0c4b': 'oo', '\u0c4c': 'au'
}
TELUGU_CONSONANTS = {
    '\u0c15': 'k',   '\u0c16': 'kh',  '\u0c17': 'g',   '\u0c18': 'gh',  '\u0c19': 'ng',
    '\u0c1a': 'c',   '\u0c1b': 'ch',  '\u0c1c': 'j',   '\u0c1d': 'jh',  '\u0c1e': 'ny',
    '\u0c1f': 'tt',  '\u0c20': 'tth', '\u0c21': 'dd',  '\u0c22': 'ddh', '\u0c23': 'nn',
    '\u0c24': 't',   '\u0c25': 'th',  '\u0c26': 'd',   '\u0c27': 'dh',  '\u0c28': 'n',
    '\u0c2a': 'p',   '\u0c2b': 'ph',  '\u0c2c': 'b',   '\u0c2d': 'bh',  '\u0c2e': 'm',
    '\u0c2f': 'y',   '\u0c30': 'r',   '\u0c31': 'rr',  '\u0c32': 'l',   '\u0c33': 'll',
    '\u0c35': 'v',   '\u0c36': 'sh',  '\u0c37': 'ss',  '\u0c38': 's',   '\u0c39': 'h'
}
VIRAMA = '\u0c4d'
ANUSVARA = '\u0c02'
VISARGA = '\u0c03'

def telugu_to_tokens(text):
    text = unicodedata.normalize('NFC', text)
    words = re.split(r'(\s+)', text)
    all_tokens = []
    for w in words:
        if not w:
            continue
        if re.match(r'^\s+$', w):
            if len(w) > 1:
                all_tokens.append('@sp')
            continue
        chars = list(w)
        i = 0
        n = len(chars)
        while i < n:
            ch = chars[i]
            if ch in TELUGU_VOWELS:
                for c in TELUGU_VOWELS[ch]:
                    all_tokens.append('@' + c)
                i += 1
            elif ch in TELUGU_CONSONANTS:
                base_c = TELUGU_CONSONANTS[ch]
                if i + 1 < n:
                    nxt = chars[i+1]
                    if nxt == VIRAMA:
                        for c in base_c:
                            all_tokens.append('@' + c)
                        i += 2
                    elif nxt in TELUGU_MATRAS:
                        for c in base_c:
                            all_tokens.append('@' + c)
                        for m in TELUGU_MATRAS[nxt]:
                            all_tokens.append('@' + m)
                        i += 2
                    else:
                        for c in base_c:
                            all_tokens.append('@' + c)
                        all_tokens.append('@a')
                        i += 1
                else:
                    for c in base_c:
                        all_tokens.append('@' + c)
                    all_tokens.append('@a')
                    i += 1
            elif ch == ANUSVARA:
                all_tokens.append('@m')
                i += 1
            elif ch == VISARGA:
                all_tokens.append('@h')
                i += 1
            else:
                i += 1
    return all_tokens

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -- Config -------------------------------------------------------------------
cfg_path = os.path.join(SCRIPT_DIR, "config.yaml")
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

stats_candidates = [
    os.path.join(SCRIPT_DIR, "stats.json"),
    os.path.join(SCRIPT_DIR, "inference", "stats.json"),
    os.path.join(BASE_DIR, "preprocessing", "Telugu", "stats.json"),
]
resolved_stats_dir = SCRIPT_DIR
for sc in stats_candidates:
    if os.path.exists(sc):
        resolved_stats_dir = os.path.dirname(sc)
        break

cfg["path"]["preprocessed_path"] = resolved_stats_dir
preprocess_config = {
    "dataset":        cfg["dataset"],
    "path":           cfg["path"],
    "preprocessing":  cfg["preprocessing"],
}
model_config = cfg["model"]

# -- Stats -------------------------------------------------------------------
stats_path = os.path.join(resolved_stats_dir, "stats.json")
with open(stats_path, "r") as f:
    stats = json.load(f)
    pitch_mean, pitch_std   = stats["pitch"][2],  stats["pitch"][3]
    energy_mean, energy_std = stats["energy"][2], stats["energy"][3]

print(f"Stats: Pitch mu={pitch_mean:.2f} sigma={pitch_std:.2f} | Energy mu={energy_mean:.2f} sigma={energy_std:.2f}")

# -- Output directories ------------------------------------------------------
RESULTS_DIR      = os.path.join(BASE_DIR, "Results", "Telugu")
AUDIO_DIR_1      = os.path.join(RESULTS_DIR, "test_generated_audios")
AUDIO_DIR_2      = os.path.join(RESULTS_DIR, "generated_audios")
PLOT_DIR         = os.path.join(RESULTS_DIR, "variance_adaptor_plots")
PLOT_DUR_DIR     = os.path.join(PLOT_DIR, "duration")
PLOT_PITCH_DIR   = os.path.join(PLOT_DIR, "pitch")
PLOT_ENERGY_DIR  = os.path.join(PLOT_DIR, "energy")
PLOT_COMB_DIR    = os.path.join(PLOT_DIR, "combined")

for d in [AUDIO_DIR_1, AUDIO_DIR_2, PLOT_DUR_DIR, PLOT_PITCH_DIR, PLOT_ENERGY_DIR, PLOT_COMB_DIR]:
    os.makedirs(d, exist_ok=True)

# -- Load FastSpeech 2 model -------------------------------------------------
model = FastSpeech2(preprocess_config, model_config).to(device)
ckpt_path = os.path.join(BASE_DIR, "Telugu", "checkpoints", "best_model.pth")
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
model.eval()
print(f"Model loaded: step {ckpt.get('step')}, best_loss {ckpt.get('best_loss')}")

# -- Load HiFi-GAN vocoder ---------------------------------------------------
vocoder = get_vocoder(model_config, device)
print(f"Vocoder loaded (native SR: {getattr(vocoder, '_native_sr', 22050)} Hz, "
      f"target SR: {cfg['preprocessing']['audio']['sampling_rate']} Hz)")

# -- Speaker -----------------------------------------------------------------
target_spk    = "telugufemale"
target_spk_id = 0
print(f"Target speaker: {target_spk}  (ID: {target_spk_id})")

cleaners      = preprocess_config["preprocessing"]["text"]["text_cleaners"]
sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
hop_length    = preprocess_config["preprocessing"]["stft"]["hop_length"]

# ----------------------------------------------------------------------------
# Telugu Test sentences
# ----------------------------------------------------------------------------
test_sentences = {
    "T1": {
        "text": "న్యూరోఫార్మకోలాజికల్ పరిశోధనలో నూతన ఔషధాల ప్రభావాన్ని విశ్లేషించారు.",
        "input_text": "న్యూరోఫార్మకోలాజికల్ పరిశోధనలో నూతన ఔషధాల ప్రభావాన్ని విశ్లేషించారు.",
        "description": "Polysyllabic technical compound: stress & duration stability across complex phonemes.",
    },
    "T2_a": {
        "text": "తాడుపై అతని పట్టు చాలా బలంగా ఉంది.",
        "input_text": "తాడుపై అతని పట్టు చాలా బలంగా ఉంది.",
        "description": "Semantic / contextual contrast: 'పట్టు' as NOUN (grip/hold).",
    },
    "T2_b": {
        "text": "ఆమె పట్టు చీరను ధరించి వేడుకకు వచ్చింది.",
        "input_text": "ఆమె పట్టు చీరను ధరించి వేడుకకు వచ్చింది.",
        "description": "Semantic / contextual contrast: 'పట్టు' as ADJECTIVE/NOUN (silk).",
    },
    "T3": {
        "text": "పులి అడ  విలో నిశ్శబ్దంగా నడిచింది.",
        "input_text": "పులి అడ  విలో నిశ్శబ్దంగా నడిచింది.",
        "description": "Orthographic split: intra-word whitespace anomaly ('అడ  విలో') inducing pause token.",
    },
    "T4": {
        "text": "ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు ప్రారంభమవుతుంది, సమయానికి సిద్ధంగా ఉంచండి.",
        "input_text": "ఈ సమావేశం రెండు వేల ఇరవై ఆరు నవంబర్ పన్నెండవ తేదీ ఉదయం పది నలభై ఐదు గంటలకు ప్రారంభమవుతుంది, సమయానికి సిద్ధంగా ఉంచండి.",
        "description": "Numbers & temporal expressions: expanded Telugu numerals (2026, 12, 10:45) and time markers.",
    },
}

parser = argparse.ArgumentParser(description="Telugu FastSpeech 2 — Assignment Test Sentence Synthesis")
parser.add_argument("--d_control", type=float, default=1.25, help="Duration control factor (default: 1.25 for calibrated clarity)")
parser.add_argument(
    "--mode",
    type=str,
    default="duration_improved",
    choices=["baseline", "duration_improved"],
    help="Duration regulation mode: 'duration_improved' (default, calibrated vowel-protected) or 'baseline'",
)
cli_args, _ = parser.parse_known_args()
print(f"Mode: {cli_args.mode} | Duration Control: {cli_args.d_control}")

synthesis_results = {}

for sent_id, info in test_sentences.items():
    print(f"\n{'='*56}")
    print(f"Synthesizing {sent_id}: {info['text'][:60]}...")
    
    tokens = telugu_to_tokens(info["input_text"])
    phone_str = "{" + " ".join(tokens) + "}"
    seq = np.array(text_to_sequence(phone_str, cleaners))

    txt_tensor = torch.from_numpy(np.array([seq])).long().to(device)
    txt_lens   = torch.from_numpy(np.array([len(seq)])).long().to(device)
    spk_tensor = torch.from_numpy(np.array([target_spk_id])).long().to(device)
    max_len    = max(txt_lens).item()

    with torch.no_grad():
        src_mask = get_mask_from_lengths(txt_lens, max_len)
        enc = model.encoder(txt_tensor, src_mask)
        if model.speaker_emb is not None:
            enc = enc + model.speaker_emb(spk_tensor).unsqueeze(1).expand(-1, max_len, -1)

        log_d = model.variance_adaptor.duration_predictor(enc, src_mask)
        d_raw = torch.clamp(torch.exp(log_d) - 1, min=0).squeeze(0)

        if cli_args.mode == "baseline":
            d_base = torch.round(d_raw)
            d_scaled = torch.clamp(torch.round(d_base * cli_args.d_control), min=0)
        else:
            d_scaled = torch.round(d_raw * cli_args.d_control)
            for idx, tok in enumerate(tokens):
                if tok in TELUGU_VOWEL_TOKENS and d_scaled[idx] < 2.0:
                    d_scaled[idx] = 2.0

        d_scaled_batch = d_scaled.unsqueeze(0)

        p_pred, pitch_emb = model.variance_adaptor.get_pitch_embedding(enc, None, src_mask, 1.0)
        enc = enc + pitch_emb
        e_pred, energy_emb = model.variance_adaptor.get_energy_embedding(enc, None, src_mask, 1.0)
        enc = enc + energy_emb

        enc, mel_len_t = model.variance_adaptor.length_regulator(enc, d_scaled_batch, None)
        mel_mask = get_mask_from_lengths(mel_len_t)
        mel_output, mel_mask = model.decoder(enc, mel_mask)
        mel_output = model.mel_linear(mel_output)
        postnet_output = model.postnet(mel_output) + mel_output
        mel_predicted = postnet_output.transpose(1, 2)

        p_pred_norm    = p_pred[0].cpu().numpy()
        e_pred_norm    = e_pred[0].cpu().numpy()
        log_d_pred     = log_d[0].cpu().numpy()
        d_rounded      = d_scaled.cpu().numpy()
        mel_len        = mel_len_t[0].item()

        lengths = mel_len_t * hop_length
        wavs    = vocoder_infer(mel_predicted, vocoder, model_config, preprocess_config, lengths=lengths)
        wav     = wavs[0]

    # Save to both audio directories
    wav_path_1 = os.path.join(AUDIO_DIR_1, f"{sent_id}.wav")
    wav_path_2 = os.path.join(AUDIO_DIR_2, f"{sent_id}.wav")
    audio_dur  = len(wav) / sampling_rate
    wavfile.write(wav_path_1, sampling_rate, wav)
    wavfile.write(wav_path_2, sampling_rate, wav)
    print(f"  Saved: {wav_path_1} and {wav_path_2} ({audio_dur:.2f} s, {mel_len} mel frames)")

    # Denormalise pitch and energy
    pitch_hz   = p_pred_norm * pitch_std  + pitch_mean
    energy_db  = e_pred_norm * energy_std + energy_mean
    dur_ms     = d_rounded   * (hop_length / sampling_rate) * 1000.0

    synthesis_results[sent_id] = {
        "text":            info["text"],
        "input_text":      info["input_text"],
        "tokens":          tokens,
        "pitch_hz":        pitch_hz.tolist(),
        "energy_db":       energy_db.tolist(),
        "duration_frames": d_rounded.tolist(),
        "duration_ms":     dur_ms.tolist(),
        "total_mel_frames": int(mel_len),
        "audio_seconds":   round(audio_dur, 3),
        "silence_pct":     round(
            float(np.sum(d_rounded[np.array(tokens) == "@sp"]))
            / max(1, d_rounded.sum()) * 100, 1
        ),
    }

    # -- Clean Standard Plots (Plain White Background, No Fancy Overlays) -----
    x_pos = np.arange(len(tokens))

    # 1. Duration
    fig, ax = plt.subplots(figsize=(max(12, len(tokens) * 0.28), 4))
    ax.bar(x_pos, dur_ms, color="#1f77b4", edgecolor="#0e4b75", alpha=0.85)
    ax.set_xticks(x_pos); ax.set_xticklabels(tokens, rotation=55, ha="right", fontsize=8)
    ax.set_xlabel("Phoneme Token"); ax.set_ylabel("Duration (ms)")
    ax.set_title(f"Predicted Phoneme Durations — {sent_id}"); ax.grid(axis="y", ls="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DUR_DIR, f"{sent_id}_duration.png"), dpi=180); plt.close()

    # 2. Pitch
    fig, ax = plt.subplots(figsize=(max(12, len(tokens) * 0.28), 4))
    ax.plot(x_pos, pitch_hz, marker="o", color="#d62728", lw=1.8, ms=3.5)
    ax.axhline(pitch_mean, color="gray", ls="--", alpha=0.7, label=f"Dataset mu = {pitch_mean:.0f} Hz")
    ax.set_xticks(x_pos); ax.set_xticklabels(tokens, rotation=55, ha="right", fontsize=8)
    ax.set_xlabel("Phoneme Token"); ax.set_ylabel("F0 (Hz)"); ax.legend()
    ax.set_title(f"Predicted Pitch Contour — {sent_id}"); ax.grid(ls="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PITCH_DIR, f"{sent_id}_pitch.png"), dpi=180); plt.close()

    # 3. Energy
    fig, ax = plt.subplots(figsize=(max(12, len(tokens) * 0.28), 4))
    ax.plot(x_pos, energy_db, marker="s", color="#2ca02c", lw=1.8, ms=3.5)
    ax.axhline(energy_mean, color="gray", ls="--", alpha=0.7, label=f"Dataset mu = {energy_mean:.0f} dB")
    ax.set_xticks(x_pos); ax.set_xticklabels(tokens, rotation=55, ha="right", fontsize=8)
    ax.set_xlabel("Phoneme Token"); ax.set_ylabel("Energy (dB)"); ax.legend()
    ax.set_title(f"Predicted Energy Contour — {sent_id}"); ax.grid(ls="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_ENERGY_DIR, f"{sent_id}_energy.png"), dpi=180); plt.close()

    # 4. Combined 3-panel
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(max(14, len(tokens) * 0.3), 10), sharex=True)
    ax1.bar(x_pos, dur_ms, color="#1f77b4", alpha=0.85); ax1.set_ylabel("Duration (ms)"); ax1.grid(axis="y", ls="--", alpha=0.4)
    ax1.set_title(f"Variance Adaptor Profiles — {sent_id}  ({info['text'][:50]}...)", fontsize=11, fontweight="bold")
    ax2.plot(x_pos, pitch_hz, marker="o", color="#d62728", lw=1.8, ms=3); ax2.axhline(pitch_mean, color="gray", ls="--", alpha=0.7)
    ax2.set_ylabel("F0 Hz"); ax2.grid(ls="--", alpha=0.4)
    ax3.plot(x_pos, energy_db, marker="s", color="#2ca02c", lw=1.8, ms=3); ax3.axhline(energy_mean, color="gray", ls="--", alpha=0.7)
    ax3.set_ylabel("Energy dB"); ax3.set_xlabel("Phoneme Token"); ax3.grid(ls="--", alpha=0.4)
    ax3.set_xticks(x_pos); ax3.set_xticklabels(tokens, rotation=55, ha="right", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_COMB_DIR, f"{sent_id}_combined.png"), dpi=180); plt.close()

    print(f"  Silence {synthesis_results[sent_id]['silence_pct']}% | "
          f"Pitch mu={pitch_hz.mean():.0f} Hz | Energy mu={energy_db.mean():.1f} dB")

# -- Save numerical summary --------------------------------------------------
summary_path = os.path.join(RESULTS_DIR, "synthesis_analysis_summary.json")
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(synthesis_results, f, indent=2, ensure_ascii=False)

print(f"\n{'='*56}")
print(f"Audio files  -> {AUDIO_DIR_1} and {AUDIO_DIR_2}")
print(f"Plots        -> {PLOT_DIR}")
print(f"JSON summary -> {summary_path}")
print("Telugu Assignment Synthesis completed successfully!")
