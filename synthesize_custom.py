
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

FASTSPEECH2_DIR = os.path.join(BASE_DIR, "FastSpeech2-master")
if FASTSPEECH2_DIR not in sys.path:
    sys.path.insert(0, FASTSPEECH2_DIR)

from model import FastSpeech2
from utils.tools import get_mask_from_lengths
import hifigan
from text import text_to_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Vowel taxonomy for selective minimum duration rule
VOWELS = {
    'a', 'aj', 'aw', 'aː', 'e', 'ej', 'eː', 'i', 'iː', 'o', 'ow', 'oː', 
    'u', 'uː', 'æ', 'ɐ', 'ɑ', 'ɑː', 'ɒ', 'ɒː', 'ɔ', 'ɔj', 'ə', 'əw', 
    'ɚ', 'ɛ', 'ɛː', 'ɜ', 'ɜː', 'ɝ', 'ɪ', 'ʉ', 'ʉː', 'ʊ'
}

def load_mfa_lexicon():
    lex_path = os.path.join(BASE_DIR, "mfa_root", "pretrained_models", "dictionary", "english_india_mfa.dict")
    lexicon = {}
    if os.path.exists(lex_path):
        with open(lex_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = re.split(r"\s+", line.strip())
                if parts and parts[0]:
                    word = parts[0].lower()
                    if word not in lexicon:
                        # Skip probability weight columns if present
                        phones = [p for p in parts[1:] if not re.match(r"^\d+\.\d+$", p)]
                        lexicon[word] = phones
    return lexicon

def text_to_phonemes(text, lexicon):
    # Normalize punctuation and tokenize
    text_clean = text.strip()
    tokens = re.findall(r"[\w']+|[.,!?;]", text_clean)
    
    phone_list = []
    for tok in tokens:
        if tok in [",", ";"]:
            phone_list.append("@sp")
        elif tok in [".", "!", "?"]:
            phone_list.append("@sp")
        else:
            w_lower = tok.lower()
            if w_lower in lexicon:
                phone_list.extend(lexicon[w_lower])
            else:
                # Character-level fallback for unknown acronyms/names
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
    args = parser.parse_args()

    # 1. Phonemize
    if args.phonemes:
        phone_str = args.phonemes
        print(f"Using provided phonemes: {phone_str}")
    else:
        lexicon = load_mfa_lexicon()
        phone_str = text_to_phonemes(args.text, lexicon)
        print(f"Input text: '{args.text}'")
        print(f"Derived phonemes: {phone_str}")

    # 2. Load model & configs
    cfg_dir = os.path.join(FASTSPEECH2_DIR, "config", "Svarah_English")
    with open(os.path.join(cfg_dir, "preprocess.yaml")) as f:
        prep_cfg = yaml.load(f, Loader=yaml.FullLoader)
    with open(os.path.join(cfg_dir, "model.yaml")) as f:
        mod_cfg = yaml.load(f, Loader=yaml.FullLoader)

    cleaners = prep_cfg["preprocessing"]["text"]["text_cleaners"]

    print("Loading HiFi-GAN Vocoder...")
    with open(os.path.join(FASTSPEECH2_DIR, "hifigan", "config.json")) as f:
        hifi_config = hifigan.AttrDict(json.load(f))
    vocoder = hifigan.Generator(hifi_config).to(device)
    hifi_ckpt = torch.load(os.path.join(FASTSPEECH2_DIR, "hifigan", "generator_universal.pth.tar"), map_location=device, weights_only=False)
    vocoder.load_state_dict(hifi_ckpt["generator"])
    vocoder.eval()
    vocoder.remove_weight_norm()

    print("Loading FastSpeech 2 Model...")
    model = FastSpeech2(prep_cfg, mod_cfg).to(device)
    ckpt = torch.load(os.path.join(BASE_DIR, "English", "checkpoints", "best_model.pth"), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # 3. Inference with Phase 4 Calibrated Duration Regulation
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
        
        # Duration prediction with post-scaling rounding
        log_d = model.variance_adaptor.duration_predictor(output, src_mask)
        d_base = torch.round(torch.clamp(torch.exp(log_d) - 1, min=0)).squeeze(0)
        d_scaled = torch.round(d_base * args.d_control)
        
        # Selective Vowel Minimum Duration Rule (>= 2 frames / 32ms)
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
        sf.write(out_path, wav_16k, 16000)
        print(f"\n[SUCCESS] Audio successfully generated: {out_path}")
        print(f"  Duration: {len(wav_16k)/16000:.2f} seconds")
        print(f"  Sample Rate: 16000 Hz")

if __name__ == "__main__":
    main()
