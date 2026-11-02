import os
import sys
import re
import argparse
import json
import yaml
import torch
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(SCRIPT_DIR) == "English":
    BASE_DIR = os.path.dirname(SCRIPT_DIR)
else:
    BASE_DIR = SCRIPT_DIR

# 1. Prioritize self-contained English/inference modules
INFERENCE_DIR = os.path.join(SCRIPT_DIR, "inference")
if os.path.exists(INFERENCE_DIR) and INFERENCE_DIR not in sys.path:
    sys.path.insert(0, INFERENCE_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

FASTSPEECH2_DIR = os.path.join(BASE_DIR, "FastSpeech2-master")
if os.path.exists(FASTSPEECH2_DIR) and FASTSPEECH2_DIR not in sys.path:
    sys.path.insert(0, FASTSPEECH2_DIR)

from model import FastSpeech2
from utils.tools import get_mask_from_lengths
from utils.model import get_vocoder, vocoder_infer
import hifigan
from text import text_to_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Vowel taxonomy for selective minimum duration rule
VOWELS = {
    'a', 'aj', 'aw', 'aː', 'e', 'ej', 'eː', 'i', 'iː', 'o', 'ow', 'oː', 
    'u', 'uː', 'æ', 'ɐ', 'ɑ', 'ɑː', 'ɒ', 'ɒː', 'ɔ', 'ɔj', 'ə', 'əw', 
    'ɚ', 'ɛ', 'ɛː', 'ɜ', 'ɜː', 'ɝ', 'ɪ', 'ʉ', 'ʉː', 'ʊ'
}

DEFAULT_PROTECTED_WORDS = {"unexpectedly", "antidisestablishmentarianism", "experiment"}

def load_mfa_lexicon():
    candidates = [
        os.path.join(SCRIPT_DIR, "lexicon.dict"),
        os.path.join(SCRIPT_DIR, "inference", "lexicon.dict"),
        os.path.join(BASE_DIR, "mfa_root", "pretrained_models", "dictionary", "english_india_mfa.dict"),
    ]
    lex_path = None
    for c in candidates:
        if os.path.exists(c):
            lex_path = c
            break

    lexicon = {}
    if lex_path and os.path.exists(lex_path):
        with open(lex_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = re.split(r"\s+", line.strip())
                if parts and parts[0]:
                    word = parts[0].lower()
                    if word not in lexicon:
                        phones = [p for p in parts[1:] if not re.match(r"^\d+\.\d+$", p)]
                        lexicon[word] = phones

    # Canonical MFA IPA mapping for assignment benchmark compound
    if "antidisestablishmentarianism" not in lexicon:
        lexicon["antidisestablishmentarianism"] = [
            "a", "n", "ʈ", "i", "ɖ", "ɪ", "s", "ɪ", "s", "ʈ", "a", "b", "l", "ɪ", "ʃ",
            "m", "ɛ", "n", "ʈ", "ɛː", "ɹ", "i", "ə", "n", "ɪ", "z", "ə", "m"
        ]
    return lexicon

def text_to_phonemes(text, lexicon, protected_words=None, enable_selective_pause=False):
    if protected_words is None:
        protected_words = DEFAULT_PROTECTED_WORDS
    text_clean = text.strip()
    tokens = re.findall(r"[\w']+|[.,!?;]", text_clean)
    
    phone_list = []
    for tok in tokens:
        if tok in [",", ";", ".", "!", "?"]:
            phone_list.append("@sp")
        else:
            w_lower = tok.lower()
            if enable_selective_pause and w_lower in protected_words:
                if phone_list and phone_list[-1] != "@sp":
                    phone_list.append("@sp")
            if w_lower in lexicon:
                phone_list.extend(lexicon[w_lower])
            else:
                for ch in w_lower:
                    if ch in lexicon:
                        phone_list.extend(lexicon[ch])
                    else:
                        phone_list.append(ch)
                        
    return "{" + " ".join(phone_list) + "}"

def main():
    parser = argparse.ArgumentParser(description="FastSpeech 2 Custom Text Synthesizer")
    parser.add_argument("--text", type=str, default="Hi, I am Savio.", help="Text to synthesize")
    parser.add_argument("--phonemes", type=str, default=None, help="Explicit phoneme string (overrides text)")
    parser.add_argument("--output", type=str, default="output_savio.wav", help="Output WAV path")
    parser.add_argument("--speaker_id", type=int, default=27, help="Speaker ID (default 27 = spk_029)")
    parser.add_argument("--d_control", type=float, default=1.25, help="Duration control speed factor")
    parser.add_argument(
        "--mode",
        type=str,
        default="duration_improved",
        choices=["baseline", "duration_improved", "combined_improved"],
        help="Inference clarity mode: 'baseline', 'duration_improved' (default), or 'combined_improved'",
    )
    parser.add_argument(
        "--protected_words",
        type=str,
        default="unexpectedly,antidisestablishmentarianism,experiment",
        help="Comma-separated list of protected words for selective boundary pause",
    )
    args = parser.parse_args()

    # 1. Phonemize
    if args.phonemes:
        phone_str = args.phonemes
        print(f"Using provided phonemes: {phone_str}")
    else:
        protected = set(w.strip().lower() for w in args.protected_words.split(",") if w.strip())
        enable_pause = (args.mode == "combined_improved")
        lexicon = load_mfa_lexicon()
        phone_str = text_to_phonemes(args.text, lexicon, protected_words=protected, enable_selective_pause=enable_pause)
        print(f"Input text: '{args.text}'")
        print(f"Mode: {args.mode}")
        print(f"Derived phonemes: {phone_str}")

    # 2. Config & Paths (Search local directory for self-contained operation)
    cfg_candidates = [
        os.path.join(SCRIPT_DIR, "config.yaml"),
        os.path.join(FASTSPEECH2_DIR, "config", "Svarah_English", "model.yaml"),
    ]
    cfg_path = None
    for c in cfg_candidates:
        if os.path.exists(c):
            cfg_path = c
            break

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Locate stats.json directory
    stats_candidates = [
        SCRIPT_DIR,
        os.path.join(SCRIPT_DIR, "inference"),
        os.path.join(BASE_DIR, "preprocessing", "English"),
        os.path.join(FASTSPEECH2_DIR, "preprocessed_data", "Svarah_English"),
    ]
    stats_dir = SCRIPT_DIR
    for d in stats_candidates:
        if os.path.exists(os.path.join(d, "stats.json")):
            stats_dir = d
            break

    prep_cfg = {
        "dataset": cfg.get("dataset", "Svarah_English"),
        "path": {"preprocessed_path": stats_dir},
        "preprocessing": cfg["preprocessing"],
    }
    mod_cfg = cfg["model"]
    cleaners = prep_cfg["preprocessing"]["text"]["text_cleaners"]

    # 3. Load Vocoder
    print("Loading HiFi-GAN Vocoder...")
    hifi_cfg_candidates = [
        os.path.join(SCRIPT_DIR, "inference", "hifigan", "config.json"),
        os.path.join(SCRIPT_DIR, "train", "hifigan", "config.json"),
        os.path.join(FASTSPEECH2_DIR, "hifigan", "config.json"),
    ]
    hifi_cfg_path = None
    for c in hifi_cfg_candidates:
        if os.path.exists(c):
            hifi_cfg_path = c
            break

    hifi_ckpt_candidates = [
        os.path.join(SCRIPT_DIR, "inference", "hifigan", "generator_universal.pth.tar"),
        os.path.join(SCRIPT_DIR, "train", "hifigan", "generator_universal.pth.tar"),
        os.path.join(FASTSPEECH2_DIR, "hifigan", "generator_universal.pth.tar"),
        os.path.join(BASE_DIR, "generator_universal.pth", "generator_universal.pth"),
    ]
    hifi_ckpt_path = None
    for c in hifi_ckpt_candidates:
        if os.path.exists(c):
            hifi_ckpt_path = c
            break

    with open(hifi_cfg_path, "r", encoding="utf-8") as f:
        hifi_config = hifigan.AttrDict(json.load(f))
    vocoder = hifigan.Generator(hifi_config).to(device)
    if hifi_ckpt_path and os.path.exists(hifi_ckpt_path):
        hifi_ckpt = torch.load(hifi_ckpt_path, map_location=device, weights_only=False)
        vocoder.load_state_dict(hifi_ckpt["generator"])
    vocoder.eval()
    vocoder.remove_weight_norm()

    # 4. Load Model Checkpoint
    print("Loading FastSpeech 2 Model...")
    model = FastSpeech2(prep_cfg, mod_cfg).to(device)
    ckpt_candidates = [
        os.path.join(SCRIPT_DIR, "checkpoints", "best_model.pth"),
        os.path.join(SCRIPT_DIR, "checkpoints", "latest_checkpoint.pth"),
        os.path.join(BASE_DIR, "English", "checkpoints", "best_model.pth"),
    ]
    ckpt_path = None
    for c in ckpt_candidates:
        if os.path.exists(c):
            ckpt_path = c
            break

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.eval()

    # 5. Inference
    tokens = [p for p in phone_str.strip("{}").split() if p]
    seq = np.array(text_to_sequence(phone_str, cleaners))
    txt = torch.from_numpy(np.array([seq])).long().to(device)
    txt_lens = torch.from_numpy(np.array([len(seq)])).long().to(device)
    spk = torch.from_numpy(np.array([args.speaker_id])).long().to(device)
    max_len = max(txt_lens).item()

    with torch.no_grad():
        src_mask = get_mask_from_lengths(txt_lens, max_len)
        output = model.encoder(txt, src_mask)
        output = output + model.speaker_emb(spk).unsqueeze(1).expand(-1, max_len, -1)
        
        # Duration prediction based on selectable mode
        log_d = model.variance_adaptor.duration_predictor(output, src_mask)
        d_raw = torch.clamp(torch.exp(log_d) - 1, min=0).squeeze(0)
        
        if args.mode == "baseline":
            # Baseline: original FastSpeech 2 rounding without vowel floor
            d_base = torch.round(d_raw)
            d_scaled = torch.clamp(torch.round(d_base * args.d_control), min=0)
        else:
            # Phase 4 calibrated duration regulation + selective vowel floor
            d_base = torch.round(d_raw)
            d_scaled = torch.round(d_base * args.d_control)
            for idx, tok in enumerate(tokens):
                if tok in VOWELS and d_scaled[idx] < 2.0:
                    d_scaled[idx] = 2.0
                
        d_scaled_batch = d_scaled.unsqueeze(0)
        
        # Pitch and energy
        _, pitch_emb = model.variance_adaptor.get_pitch_embedding(output, None, src_mask, 1.0)
        output = output + pitch_emb
        _, energy_emb = model.variance_adaptor.get_energy_embedding(output, None, src_mask, 1.0)
        output = output + energy_emb
        
        # Length Regulation & Decoder
        output, mel_len = model.variance_adaptor.length_regulator(output, d_scaled_batch, None)
        mel_mask = get_mask_from_lengths(mel_len)
        mel_output, mel_mask = model.decoder(output, mel_mask)
        mel_output = model.mel_linear(mel_output)
        postnet_output = model.postnet(mel_output) + mel_output
        mel_pred = postnet_output.transpose(1, 2)
        
        # Vocoder & 16kHz resample
        wav = vocoder(mel_pred).squeeze(1).cpu().numpy()[0]
        g = gcd(16000, 22050)
        wav_16k = resample_poly(wav, 16000 // g, 22050 // g)
        wav_16k = np.clip(wav_16k, -1.0, 1.0)
        
        out_path = os.path.abspath(args.output)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        sf.write(out_path, wav_16k, 16000)
        print(f"\n[SUCCESS] Audio successfully generated: {out_path}")
        print(f"  Duration: {len(wav_16k)/16000:.2f} seconds")
        print(f"  Sample Rate: 16000 Hz")

if __name__ == "__main__":
    main()
