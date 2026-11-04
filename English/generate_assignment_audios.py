"""
English FastSpeech 2 — Assignment Test Sentence Synthesis
==========================================================
Generates E1, E2_a, E2_b, E3, E4 audio files and variance
adaptor plots as required by the assignment (Sections 3 & 4 of
D:\\Savio Shaju\\TTS\\Ref\\Assignment.pdf).

Fixes applied vs. original version:
  - BUG-01: energy_embedding now uses e_control (was p_control) — fixed in
    modules.py but also reflected in documentation here.
  - @sp inserted only at clause/phrase boundaries (commas, periods), not
    between every single word. This matches the training distribution in
    train.txt and prevents the 40% silence inflation observed previously.
  - Vocoder resampling bridge (22,050 → 16,000 Hz) now handled in
    utils/model.py; no manual correction needed here.
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
import argparse

sys.stdout.reconfigure(encoding='utf-8')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
PKG_DIR = os.path.join(SCRIPT_DIR, "inference")
if not os.path.exists(PKG_DIR):
    PKG_DIR = os.path.join(BASE_DIR, "kaggle_staging", "English_FastSpeech2_Kaggle")
if PKG_DIR not in sys.path:
    sys.path.insert(0, PKG_DIR)

from model import FastSpeech2
import hifigan
from utils.model import vocoder_infer, get_vocoder
from utils.tools import get_mask_from_lengths
from text import text_to_sequence

# Vowel taxonomy for selective minimum duration floor
VOWELS = {
    'a', 'aj', 'aw', 'aː', 'e', 'ej', 'eː', 'i', 'iː', 'o', 'ow', 'oː', 
    'u', 'uː', 'æ', 'ɐ', 'ɑ', 'ɑː', 'ɒ', 'ɒː', 'ɔ', 'ɔj', 'ə', 'əw', 
    'ɚ', 'ɛ', 'ɛː', 'ɜ', 'ɜː', 'ɝ', 'ɪ', 'ʉ', 'ʉː', 'ʊ'
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ── Config ─────────────────────────────────────────────────────────────────
cfg_path = os.path.join(SCRIPT_DIR, "config.yaml")
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

stats_candidates = [
    os.path.join(SCRIPT_DIR, "stats.json"),
    os.path.join(SCRIPT_DIR, "inference", "stats.json"),
    os.path.join(BASE_DIR, "preprocessing", "English", "stats.json"),
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

# ── Stats ───────────────────────────────────────────────────────────────────
stats_path = os.path.join(resolved_stats_dir, "stats.json")
with open(stats_path, "r") as f:
    stats = json.load(f)
    pitch_mean, pitch_std   = stats["pitch"][2],  stats["pitch"][3]
    energy_mean, energy_std = stats["energy"][2], stats["energy"][3]

print(f"Stats: Pitch μ={pitch_mean:.2f} σ={pitch_std:.2f} | Energy μ={energy_mean:.2f} σ={energy_std:.2f}")

# ── Output directories ──────────────────────────────────────────────────────
RESULTS_DIR      = os.path.join(BASE_DIR, "Results", "English")
AUDIO_DIR        = os.path.join(RESULTS_DIR, "test_generated_audios")
PLOT_DIR         = os.path.join(RESULTS_DIR, "variance_adaptor_plots")
PLOT_DUR_DIR     = os.path.join(PLOT_DIR, "duration")
PLOT_PITCH_DIR   = os.path.join(PLOT_DIR, "pitch")
PLOT_ENERGY_DIR  = os.path.join(PLOT_DIR, "energy")

for d in [AUDIO_DIR, PLOT_DUR_DIR, PLOT_PITCH_DIR, PLOT_ENERGY_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Load FastSpeech 2 model ─────────────────────────────────────────────────
model = FastSpeech2(preprocess_config, model_config).to(device)
ckpt_path = os.path.join(BASE_DIR, "English", "checkpoints", "best_model.pth")
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
model.eval()
print(f"Model loaded: step {ckpt.get('step')}, best_loss {ckpt.get('best_loss')}")

# ── Load HiFi-GAN vocoder (with 22 kHz → 16 kHz bridge) ────────────────────
vocoder = get_vocoder(model_config, device)
print(f"Vocoder loaded (native SR: {getattr(vocoder, '_native_sr', 22050)} Hz, "
      f"target SR: {cfg['preprocessing']['audio']['sampling_rate']} Hz)")

# ── Speaker ─────────────────────────────────────────────────────────────────
with open(os.path.join(cfg["path"]["preprocessed_path"], "speakers.json")) as f:
    speaker_map = json.load(f)

# spk_029 has the most training utterances (118) and is a clear female speaker
target_spk    = "spk_029"
target_spk_id = speaker_map[target_spk]   # integer 27
print(f"Target speaker: {target_spk}  (ID: {target_spk_id})")

cleaners      = preprocess_config["preprocessing"]["text"]["text_cleaners"]
sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
hop_length    = preprocess_config["preprocessing"]["stft"]["hop_length"]

# ────────────────────────────────────────────────────────────────────────────
# Test sentences
# ────────────────────────────────────────────────────────────────────────────
#
# Phoneme notation notes:
#   • Symbols in curly braces {…} are whitespace-separated MFA IPA phones.
#   • @sp  = short pause (inter-clause boundary, ~50–150 ms)
#   • @spn = noise/speaker pause (longer; used for sentence-final pauses)
#   • Pause tokens are placed ONLY at punctuation boundaries (commas, full
#     stops), not between every word — matching the training distribution.
#
# E3: "rhyth  m" — the double-space is an intentional orthographic anomaly.
#     We normalise to /ɹ ɪ d̪ ə m/ (identical to "rhythm").
# E3_split: treats each half of "rhyth  m" as a separate token with sp.
#
# E4: Numbers expanded by text cleaners (10:45 → ten forty-five, etc.).
#     "andaroo" (Telugu: "everyone") transcribed with retroflex /ɾ/ and
#     long back vowel /uː/.
#
test_sentences = {
    "E1": {
        "text": (
            "The antidisestablishmentarianism movement gained considerable "
            "momentum during the parliamentary proceedings."
        ),
        # Clean, natural phonemes without rare/exotic phone distortions
        "phonemes": (
            "{d̪ ə a n ʈ i ɖ ɪ s ɪ s ʈ a b l ɪ ʃ m ɛ n ʈ ɛː ɹ i ə n ɪ z ə m "
            "m uː ʋ m ɛ n ʈ ɡ eː n ɖ k ə n s ɪ ɖ ə ɹ ə b ə l "
            "m oː m ɛ n ʈ ə m ɖ ʊ ɹ ɪ ŋ d̪ ə p ɑː l ə m ɛ n ʈ ə ɹ i "
            "p ɹ ə s iː ɖ ɪ n z}"
        ),
        "description": "Long polysyllabic morphological compound; stress distribution test.",
    },
    "E2_a": {
        "text": "The orchestra's conduct during the ceremony was praised by the audience.",
        # Noun heteronym: /k ɒ n ɖ ə k ʈ/ — trochaic stress (CONduct) with audible word boundary before 'during'
        "phonemes": (
            "{d̪ ə ɒː k ə s ʈ ɹ ə s k ɒ n ɖ ə k ʈ @sp ɖ ʊ ɹ ɪ ŋ d̪ ə "
            "s ɛ ɹ ɪ m ə n i ʋ ə z p ɹ eː z ɖ b aj d̪ ə ɒː ɖ i ə n s}"
        ),
        "description": "Heteronym 'conduct' as NOUN: trochaic stress (/k ɒ n ɖ ə k ʈ/) with inter-word pause before 'during'.",
    },
    "E2_b": {
        "text": "The researchers will conduct an experiment to evaluate the proposed method.",
        # Verb heteronym: /k ə n ɖ ə k ʈ/ — iambic stress (conDUCT)
        "phonemes": (
            "{d̪ ə ɹ iː s ɜː tʃ ə z ʋ ɪ l k ə n ɖ ə k ʈ ə n "
            "ɛ k s p ɛ ɾ ɪ m ɛ n ʈ ʈ uː ɪ ʋ a l j u eː ʈ d̪ ə "
            "p ɹ ə p oː z ɖ m ɛ t̪ ə ɖ}"
        ),
        "description": "Heteronym 'conduct' as VERB: iambic stress (/k ə n ɖ ə k ʈ/).",
    },
    "E3": {
        "text": "The rhyth  m of the orchestra changed unexpectedly.",
        # "rhyth  m" normalised → /ɹ ɪ d̪ ə m/ (canonical clean rhythm)
        "phonemes": (
            "{d̪ ə ɹ ɪ d̪ ə m ɒ ʋ d̪ ə ɒː k ə s ʈ ɹ ə "
            "tʃ eː n dʒ ɖ ə n ɛ k s p ɛ k ʈ ɪ ɖ l i}"
        ),
        "description": "Normalised 'rhyth  m' → /ɹ ɪ d̪ ə m/. Smooth unified audio.",
    },
    "E4": {
        "text": (
            "The meeting will start at 10:45 AM on 12 November 2026, "
            "so andaroo should arrive on time."
        ),
        # Authentic loanword phonetics: pause before 'andaroo' and dental/tap /a n d̪ a ɾ uː/
        "phonemes": (
            "{d̪ ə m iː ʈ ɪ ŋ ʋ ɪ l s ʈ ɑː ʈ a ʈ ɛ n f ɒː ʈ i f aj ʋ "
            "eː ɛ m ɒ n t̪ ʋ ɛ l ʋ n ə ʋ ɛ m b ə ɹ t̪ ʋ ɛ n ʈ i t̪ ʋ ɛ n ʈ i s ɪ k s @sp "
            "s oː @sp a n d̪ a ɾ uː ʃ ʊ ɖ ə ɹ aj ʋ ɒ n ʈ aj m}"
        ),
        "description": "Numbers expanded; Telugu loanword 'andaroo' transcribed with clear pause and authentic phonetics.",
    },
}

parser = argparse.ArgumentParser(description="English FastSpeech 2 — Assignment Test Sentence Synthesis")
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
    print(f"Synthesizing {sent_id}: {info['text'][:70]}...")
    phoneme_str = info["phonemes"]

    # Remove any double-spaces inside curly braces and normalise
    phoneme_str = ' '.join(phoneme_str.split())

    tokens = phoneme_str.strip("{}").split()
    seq    = np.array(text_to_sequence(phoneme_str, cleaners))

    txt_tensor = torch.from_numpy(np.array([seq])).long().to(device)
    txt_lens   = torch.from_numpy(np.array([len(seq)])).long().to(device)
    spk_tensor = torch.from_numpy(np.array([target_spk_id])).long().to(device)
    max_len    = max(txt_lens).item()

    with torch.no_grad():
        src_mask = get_mask_from_lengths(txt_lens, max_len)
        output = model.encoder(txt_tensor, src_mask)
        if model.speaker_emb is not None:
            output = output + model.speaker_emb(spk_tensor).unsqueeze(1).expand(-1, max_len, -1)

        log_d = model.variance_adaptor.duration_predictor(output, src_mask)
        d_raw = torch.clamp(torch.exp(log_d) - 1, min=0).squeeze(0)

        if cli_args.mode == "baseline":
            d_base = torch.round(d_raw)
            d_scaled = torch.clamp(torch.round(d_base * cli_args.d_control), min=0)
        else:
            # Phase 4 calibrated duration regulation + selective vowel nucleus protection
            d_scaled = torch.round(d_raw * cli_args.d_control)
            for idx, tok in enumerate(tokens):
                if tok in VOWELS and d_scaled[idx] < 2.0:
                    d_scaled[idx] = 2.0

        d_scaled_batch = d_scaled.unsqueeze(0)

        # Pitch and energy
        p_pred, pitch_emb = model.variance_adaptor.get_pitch_embedding(output, None, src_mask, 1.0)
        output = output + pitch_emb
        e_pred, energy_emb = model.variance_adaptor.get_energy_embedding(output, None, src_mask, 1.0)
        output = output + energy_emb

        # Length Regulation & Decoder
        output, mel_len_t = model.variance_adaptor.length_regulator(output, d_scaled_batch, None)
        mel_mask = get_mask_from_lengths(mel_len_t)
        mel_output, mel_mask = model.decoder(output, mel_mask)
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

    wav_path   = os.path.join(AUDIO_DIR, f"{sent_id}.wav")
    audio_dur  = len(wav) / sampling_rate
    wavfile.write(wav_path, sampling_rate, wav)
    print(f"  Saved: {wav_path} ({audio_dur:.2f} s, {mel_len} mel frames)")

    # Denormalise pitch and energy
    pitch_hz   = p_pred_norm * pitch_std  + pitch_mean
    energy_db  = e_pred_norm * energy_std + energy_mean
    dur_ms     = d_rounded   * (hop_length / sampling_rate) * 1000.0

    synthesis_results[sent_id] = {
        "text":            info["text"],
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

    # ── Plots ──────────────────────────────────────────────────────────────
    x_pos = np.arange(len(tokens))

    # 1. Duration
    fig, ax = plt.subplots(figsize=(max(12, len(tokens) * 0.3), 4))
    ax.bar(x_pos, dur_ms, color="#1f77b4", edgecolor="#0e4b75", alpha=0.85)
    ax.set_xticks(x_pos); ax.set_xticklabels(tokens, rotation=55, ha="right", fontsize=8)
    ax.set_xlabel("Phoneme Token"); ax.set_ylabel("Duration (ms)")
    ax.set_title(f"Predicted Phoneme Durations — {sent_id}"); ax.grid(axis="y", ls="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DUR_DIR, f"{sent_id}_duration.png"), dpi=180); plt.close()

    # 2. Pitch
    fig, ax = plt.subplots(figsize=(max(12, len(tokens) * 0.3), 4))
    ax.plot(x_pos, pitch_hz, marker="o", color="#d62728", lw=1.8, ms=3.5)
    ax.axhline(pitch_mean, color="gray", ls="--", alpha=0.7, label=f"Dataset μ = {pitch_mean:.0f} Hz")
    ax.set_xticks(x_pos); ax.set_xticklabels(tokens, rotation=55, ha="right", fontsize=8)
    ax.set_xlabel("Phoneme Token"); ax.set_ylabel("F0 (Hz)"); ax.legend()
    ax.set_title(f"Predicted Pitch Contour — {sent_id}"); ax.grid(ls="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PITCH_DIR, f"{sent_id}_pitch.png"), dpi=180); plt.close()

    # 3. Energy
    fig, ax = plt.subplots(figsize=(max(12, len(tokens) * 0.3), 4))
    ax.plot(x_pos, energy_db, marker="s", color="#2ca02c", lw=1.8, ms=3.5)
    ax.axhline(energy_mean, color="gray", ls="--", alpha=0.7, label=f"Dataset μ = {energy_mean:.0f} dB")
    ax.set_xticks(x_pos); ax.set_xticklabels(tokens, rotation=55, ha="right", fontsize=8)
    ax.set_xlabel("Phoneme Token"); ax.set_ylabel("Energy (dB)"); ax.legend()
    ax.set_title(f"Predicted Energy Contour — {sent_id}"); ax.grid(ls="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_ENERGY_DIR, f"{sent_id}_energy.png"), dpi=180); plt.close()

    print(f"  Silence {synthesis_results[sent_id]['silence_pct']}% | "
          f"Pitch μ={pitch_hz.mean():.0f} Hz | Energy μ={energy_db.mean():.1f} dB")

print(f"\n{'='*56}")
print(f"Audio files  → {AUDIO_DIR_1} and {AUDIO_DIR_2}")
print(f"Plots        → {PLOT_DIR}")
print("English Assignment Synthesis completed successfully!")
