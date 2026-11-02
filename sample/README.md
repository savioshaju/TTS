# FastSpeech 2 Audio Samples (English & Telugu)

This directory contains demonstration speech audio synthesized using our trained **FastSpeech 2** models paired with pre-trained **HiFi-GAN** vocoders. All samples are generated with calibrated duration scaling (`--mode duration_improved --d_control 1.25`) to prevent word clipping and ensure natural sentence rhythm.

---

## Directory Structure

```
sample/
├── English/
│   ├── 01_savio_greeting.wav              # Self-introduction & conversational greeting
│   ├── 02_orchestra_clarity.wav           # Suffix & plosive clarity test
│   ├── 03_fastspeech_overview.wav         # Technical explanation with polysyllabic terms
│   ├── 04_antidisestablishmentarianism.wav# 28-letter polysyllabic compound word test
│   └── 05_meeting_schedule.wav            # Temporal expressions & numeral-containing sentence
├── Telugu/
│   ├── 01_telugu_greeting.wav             # Standard respectful greeting & project intro
│   ├── 02_weather_sentence.wav            # Conversational weather evaluation
│   ├── 03_technical_compound.wav          # High-complexity compound Sanskrit-origin loan words
│   ├── 04_pattu_saree.wav                 # Cultural contextual sentence with geminates
│   └── 05_meeting_schedule.wav            # Formal scheduled announcement with numbers & dates
├── generate_samples.py                    # Automated sample generation script
└── README.md                              # This documentation
```

---

## 1. English Audio Samples

Synthesized using the **AI4Bharat Svarah** multi-speaker model with our expanded phonetic dictionary (incorporating rule-based G2P and acronym preservation):

| File | Transcript / Prompt | Focus & Key Phonetic Traits |
| :--- | :--- | :--- |
| **`01_savio_greeting.wav`** | *"Hi, I am Savio. Welcome to our speech synthesis demonstration."* | Demonstrates name synthesis (`Savio` as `/s a ʋ i oː/` rather than spelled out) and natural conversational flow. |
| **`02_orchestra_clarity.wav`** | *"The orchestra changed unexpectedly."* | Evaluates word-boundary preservation before vowel-initial words (`unexpectedly`) and plosive clusters. |
| **`03_fastspeech_overview.wav`** | *"FastSpeech 2 synthesizes speech non-autoregressively with high speed and natural prosody."* | Evaluates pronunciation of technical terms, hyphenated adverbs, and multi-syllabic jargon. |
| **`04_antidisestablishmentarianism.wav`** | *"The antidisestablishmentarianism movement gained considerable momentum."* | Stress test on a 28-letter, 12-syllable word. Demonstrates preserved phoneme duration without syllable dropouts. |
| **`05_meeting_schedule.wav`** | *"The meeting will start at 10:45 AM on 12 November 2026."* | Tests mixed temporal expressions, acronyms (`AM`), and numerical prosodic phrasing. |

---

## 2. Telugu Audio Samples (తెలుగు నమూనాలు)

Synthesized using the **IndicTTS** bilingual Telugu female voice model with vowel nucleus duration floors ($\ge 2$ frames) and calibrated speed regulation:

| File | Telugu Transcript | English Translation | Key Phonetic Traits |
| :--- | :--- | :--- | :--- |
| **`01_telugu_greeting.wav`** | నమస్కారం, మీరు ఎలా ఉన్నారు? మా తెలుగు స్పీచ్ సింథసిస్ ప్రాజెక్ట్‌కి స్వాగతం. | *Hello, how are you? Welcome to our Telugu speech synthesis project.* | Polite formal cadence, conjunct consonant clusters (`స్పీచ్`, `ప్రాజెక్ట్`). |
| **`02_weather_sentence.wav`** | ఈ రోజు వాతావరణం చాలా బాగుంది. | *The weather today is very pleasant.* | Everyday conversational Telugu; evaluates vowel length distinction between short and long vowels (`చాలా`, `బాగుంది`). |
| **`03_technical_compound.wav`** | న్యూరోఫార్మకోలాజికల్ పరిశోధనలో నూతన ఔషధాల ప్రభావాన్ని విశ్లేషించారు. | *The efficacy of novel drugs was analyzed in neuropharmacological research.* | Complex scientific compound loan-words and aspirates (`ప్రభావం`, `ఔషధం`, `విశ్లేషణ`). |
| **`04_pattu_saree.wav`** | ఆమె పట్టు చీరను ధరించి వేడుకకు వచ్చింది. | *She wore a silk saree and attended the celebration.* | Geminate consonant duration (`పట్టు`, `వచ్చింది`) and cultural vocabulary cadence. |
| **`05_meeting_schedule.wav`** | ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు ప్రారంభమవుతుంది, సమయానికి సిద్ధంగా ఉంచండి. | *This meeting will start on 12 November 2026 at 10:45 AM, please be ready on time.* | Long-form complex announcement testing multi-clause breath pauses and temporal numbers. |

---

## 3. How to Regenerate or Generate New Audios

### Regenerate All Samples:
```bash
python sample/generate_samples.py
```

### Synthesize Custom English Sentences:
```bash
python English/synthesize_custom.py --text "Your English text here." --d_control 1.25 --mode duration_improved --output custom_output.wav
```

### Synthesize Custom Telugu Sentences:
```bash
python Telugu/synthesize_custom.py --text "మీ తెలుగు వాక్యం ఇక్కడ రాయండి." --d_control 1.25 --mode duration_improved --output custom_te.wav
```

