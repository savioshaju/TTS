"""
Generate FastSpeech 2 Audio Samples for English and Telugu
==========================================================
Generates high-quality speech samples using our trained FastSpeech 2
models and HiFi-GAN vocoders with duration improvements.
"""

import os
import sys
import subprocess

# Ensure UTF-8 output encoding on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

EN_DIR = os.path.join(SCRIPT_DIR, "English")
TE_DIR = os.path.join(SCRIPT_DIR, "Telugu")
os.makedirs(EN_DIR, exist_ok=True)
os.makedirs(TE_DIR, exist_ok=True)

ENGLISH_SAMPLES = [
    (
        "01_savio_greeting.wav",
        "Hi, I am Savio. Welcome to our speech synthesis demonstration.",
    ),
    (
        "02_orchestra_clarity.wav",
        "The orchestra changed unexpectedly.",
    ),
    (
        "03_fastspeech_overview.wav",
        "FastSpeech 2 synthesizes speech non-autoregressively with high speed and natural prosody.",
    ),
    (
        "04_antidisestablishmentarianism.wav",
        "The antidisestablishmentarianism movement gained considerable momentum.",
    ),
    (
        "05_meeting_schedule.wav",
        "The meeting will start at 10:45 AM on 12 November 2026.",
    ),
]

TELUGU_SAMPLES = [
    (
        "01_telugu_greeting.wav",
        "నమస్కారం, మీరు ఎలా ఉన్నారు? మా తెలుగు స్పీచ్ సింథసిస్ ప్రాజెక్ట్‌కి స్వాగతం.",
    ),
    (
        "02_weather_sentence.wav",
        "ఈ రోజు వాతావరణం చాలా బాగుంది.",
    ),
    (
        "03_technical_compound.wav",
        "న్యూరోఫార్మకోలాజికల్ పరిశోధనలో నూతన ఔషధాల ప్రభావాన్ని విశ్లేషించారు.",
    ),
    (
        "04_pattu_saree.wav",
        "ఆమె పట్టు చీరను ధరించి వేడుకకు వచ్చింది.",
    ),
    (
        "05_meeting_schedule.wav",
        "ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు ప్రారంభమవుతుంది, సమయానికి సిద్ధంగా ఉంచండి.",
    ),
]

def generate_english():
    print("=" * 60)
    print("Generating English FastSpeech 2 Samples...")
    print("=" * 60)
    en_script = os.path.join(BASE_DIR, "English", "synthesize_custom.py")
    for filename, text in ENGLISH_SAMPLES:
        out_path = os.path.join(EN_DIR, filename)
        if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
            print(f"  -> Skipping existing: {filename} ({os.path.getsize(out_path):,} bytes)")
            continue
        print(f"\n[EN] Synthesizing: '{text}' -> {filename}")
        cmd = [
            sys.executable,
            en_script,
            "--text", text,
            "--d_control", "1.25",
            "--mode", "duration_improved",
            "--output", out_path,
        ]
        res = subprocess.run(cmd, cwd=BASE_DIR, capture_output=True, text=True, encoding="utf-8")
        if res.returncode == 0:
            print(f"  -> SUCCESS: {filename}")
        else:
            print(f"  -> ERROR:\n{res.stderr}")

def generate_telugu():
    print("\n" + "=" * 60)
    print("Generating Telugu FastSpeech 2 Samples...")
    print("=" * 60)
    te_script = os.path.join(BASE_DIR, "Telugu", "synthesize_custom.py")
    for filename, text in TELUGU_SAMPLES:
        out_path = os.path.join(TE_DIR, filename)
        if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
            print(f"  -> Skipping existing: {filename} ({os.path.getsize(out_path):,} bytes)")
            continue
        print(f"\n[TE] Synthesizing: '{filename}' -> {filename}")
        cmd = [
            sys.executable,
            te_script,
            "--text", text,
            "--d_control", "1.25",
            "--mode", "duration_improved",
            "--output", out_path,
        ]
        res = subprocess.run(cmd, cwd=BASE_DIR, capture_output=True, text=True, encoding="utf-8")
        if res.returncode == 0:
            print(f"  -> SUCCESS: {filename}")
        else:
            print(f"  -> ERROR:\n{res.stderr}")

if __name__ == "__main__":
    generate_english()
    generate_telugu()
    print("\nAll FastSpeech 2 samples generated successfully in 'sample/' directory!")

