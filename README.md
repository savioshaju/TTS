# Implementation of FastSpeech 2-Based Text-to-Speech Synthesis
## English (AI4Bharat Svarah) & Telugu (IndicTTS)

A complete, production-grade implementation of the **FastSpeech 2** non-autoregressive text-to-speech architecture developed separately for **Indian English** and **Telugu**, fulfilling all requirements specified in the assignment specification.

---

## Table of Contents
1. [Objective & Project Overview](#1-objective--project-overview)
2. [Deliverables & Repository Layout](#2-deliverables--repository-layout)
3. [Prerequisites and Environment Setup](#3-prerequisites-and-environment-setup)
4. [Model Checkpoints & Vocoders (Google Drive)](#4-model-checkpoints--vocoders-google-drive)
5. [Dataset Download Instructions (Hugging Face CLI)](#5-dataset-download-instructions-hugging-face-cli)
6. [5-Hour Dataset Preparation and Subsampling](#6-5-hour-dataset-preparation-and-subsampling)
7. [Audio Standardization, Forced Alignment & Acoustic Preprocessing](#7-audio-standardization-forced-alignment--acoustic-preprocessing)
8. [Training Instructions (Kaggle Dual-GPU & Local)](#8-training-instructions-kaggle-dual-gpu--local)
9. [Inference & Speech Synthesis (Benchmark & Custom)](#9-inference--speech-synthesis-benchmark--custom)
10. [Benchmark Evaluation & Linguistic Results](#10-benchmark-evaluation--linguistic-results)
11. [Loss Curves & Convergence Analysis](#11-loss-curves--convergence-analysis)

---

## 1. Objective & Project Overview

The objective of this assignment is to implement, train, evaluate, and comparatively analyze the **FastSpeech 2** Text-to-Speech (TTS) model independently for two distinct languages:
- **Indian English (AI4Bharat Svarah Corpus)**: 104 speakers across 19 Indian native-language backgrounds, 100 IPA phoneme tokens with Montreal Forced Aligner (MFA v2.2).
- **Telugu (IndicTTS Corpus)**: Bilingual studio recordings (1 female, 1 male), 460 native *akshara* tokens with Meta MMS CTC forced alignment.

### Key Architecture Components
* **Feed-Forward Transformer (FFT)**: 4-layer encoder and 6-layer decoder with multi-head self-attention (2 heads, $d_{\text{model}}=256$) and 1D convolutions.
* **Variance Adaptor**: Explicitly predicts duration, fundamental frequency ($F_0$), and frame energy ($E$), injecting fine-grained prosody directly into expanded representations.
* **Vocoder Adaptation**: Universal HiFi-GAN vocoder paired with polyphase Kaiser sinc-resampling (22.05 kHz $\to$ 16.0 kHz) and calibrated duration scaling ($d_{\text{control}} \approx 1.38 - 1.40$) for natural conversational cadence.

---

## 2. Deliverables & Repository Layout

The repository strictly conforms to the directory structure required in **Section 4.a** of the assignment:

```text
project/
├── English/
│   ├── train.py                     # Dual-GPU / local training engine
│   ├── preprocess.py                # Svarah MFA alignment & feature extraction
│   ├── inference.py                 # Single-sentence CLI & benchmark synthesis
│   ├── config.yaml                  # Model, acoustic, and training hyperparameters
│   ├── checkpoints/                 # Checkpoint directory (best_model.pth)
│   ├── stats.json                   # Pitch & energy z-score normalization statistics
│   ├── speakers.json                # 104 speaker demographic mapping
│   ├── lexicon.dict                 # MFA IPA pronunciation dictionary
│   └── English_FastSpeech2_Kaggle.ipynb # Dual-GPU Kaggle training notebook
│
├── Telugu/
│   ├── train.py                     # Dual-GPU / local training engine
│   ├── preprocess.py                # IndicTTS MMS alignment & akshara extraction
│   ├── inference.py                 # Single-sentence CLI & benchmark synthesis
│   ├── config.yaml                  # Model, acoustic, and training hyperparameters
│   ├── checkpoints/                 # Checkpoint directory (best_model.pth)
│   ├── stats.json                   # Pitch & energy z-score normalization statistics
│   ├── speakers.json                # 2 studio speaker mapping
│   └── Telugu_FastSpeech2_Kaggle.ipynb # Dual-GPU Kaggle training notebook
│
├── Results/
│   ├── English/
│   │   ├── training_loss.png        # Training loss curve (100,000 steps)
│   │   ├── validation_loss.png      # Validation loss curve (100,000 steps)
│   │   ├── test_generated_audios/   # Generated benchmark audios (E1.wav to E4.wav)
│   │   └── variance_adaptor_plots/
│   │       ├── duration/            # Duration predictor alignment plots
│   │       ├── pitch/               # F0 pitch contour tracking plots
│   │       ├── energy/              # Energy contour tracking plots
│   │       └── combined/            # 3-in-1 consolidated profile figures
│   └── Telugu/
│       ├── training_loss.png        # Training loss curve (100,000 steps)
│       ├── validation_loss.png      # Validation loss curve (100,000 steps)
│       ├── test_generated_audios/   # Generated benchmark audios (T1.wav to T4.wav)
│       └── variance_adaptor_plots/
│           ├── duration/            # Duration predictor alignment plots
│           ├── pitch/               # F0 pitch contour tracking plots
│           ├── energy/              # Energy contour tracking plots
│           └── combined/            # 3-in-1 consolidated profile figures
│
├── Report/
│   └── Technical_Report.pdf         # Comprehensive formal technical report (compiled XeLaTeX)
├── report_source/                   # LaTeX source files and raw figures
│   ├── Technical_Report.tex
│   └── figures/
├── README.md                        # Documentation, download & reproduction guide
└── requirements.txt                 # Exact pinned dependencies
```

---

## 3. Prerequisites and Environment Setup

To reproduce this project from scratch, follow these setup steps in order:

### 3.1. Step 1: Clone the Repository
Clone the project repository from GitHub and navigate into the workspace directory:
```bash
git clone https://github.com/savioshaju/TTS.git
cd TTS
```

### 3.2. Step 2: Virtual Environment Creation
Use Python 3.11 to create and activate a virtual environment:
```bash
python -m venv .venv

# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (Command Prompt):
.venv\Scripts\activate.bat
# Linux / macOS:
source .venv/bin/activate
```

### 3.3. Step 3: Install Dependencies
Upgrade pip and install all pinned packages from [`requirements.txt`](requirements.txt):
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```
The implementation runs on Python 3.11 with PyTorch 2.5, TorchAudio, NumPy, SciPy, Librosa, PyWorld, Matplotlib, and HuggingFace Transformers. Pinned package versions are listed in [`requirements.txt`](requirements.txt).

### 3.4. External Alignment Tools
* **English (Montreal Forced Aligner v2.2)**: Install MFA in a dedicated conda environment and download the pre-trained acoustic model and Indian English dictionary:
  ```bash
  # Create and activate conda environment for MFA
  conda create -n mfa python=3.10 -y
  conda activate mfa
  conda install -c conda-forge montreal-forced-aligner=2.2.17 -y

  # Download acoustic model and Indian English dictionary
  mfa model download acoustic english_mfa
  mfa model download dictionary english_india_mfa
  ```
* **Telugu (Meta MMS Aligner)**: Telugu forced alignment uses Meta's Massively Multilingual Speech model (`facebook/mms-1b-all`). Model weights download automatically via HuggingFace Transformers on the first preprocessing run; no external tools or compilers are required.

---

## 4. Model Checkpoints & Vocoders (Google Drive)

> [!IMPORTANT]
> **Checkpoints Download Link (Google Drive)**:  
> 🔗 **[Google Drive Shared Checkpoints Folder](https://drive.google.com/drive/folders/1MYS5zcsyehxDN5n4XaKj5jCidpUZ-efw?usp=sharing)**  
> *Per Assignment Section 4.b, model checkpoints exceeding GitHub's file-size limits are hosted on Google Drive. The link is publicly accessible and permanent.*

### Included Assets & Checkpoint Downloads:
1. **HiFi-GAN Neural Vocoders (`generator_universal.pth.tar`)**: Included directly in the repository at:
   - `English/inference/hifigan/generator_universal.pth.tar`
   - `Telugu/inference/hifigan/generator_universal.pth.tar`
   *(No separate vocoder download required)*
2. **FastSpeech 2 Model Checkpoints**: Download **only 2 files** from Google Drive and place them into the respective `checkpoints/` directories:

| Language | Checkpoint in Drive | Local Destination Path | File Size | Checkpoint Step & Description |
| :--- | :--- | :--- | :---: | :--- |
| **English** | `English/best_model.pth` | `English/checkpoints/best_model.pth` | ~399 MB | Step 61,000 (Optimal Perceptual Quality) |
| **Telugu** | `Telugu/best_model.pth` | `Telugu/checkpoints/best_model.pth` | ~400 MB | Step 74,000 (Optimal Perceptual Quality) |

### Checkpoint Verification Command
Once copied, verify file placement:

**PowerShell (Windows):**
```powershell
Test-Path English/checkpoints/best_model.pth, Telugu/checkpoints/best_model.pth
# Expected output: True, True
```

**Bash (Linux / macOS):**
```bash
ls -lh English/checkpoints/best_model.pth Telugu/checkpoints/best_model.pth
```

---

## 5. Dataset Download Instructions (Hugging Face CLI)

Download the raw datasets directly from Hugging Face using the official Hugging Face Hub CLI (zero custom download scripts required):

```bash
# Install Hugging Face Hub CLI
pip install huggingface_hub

# English: AI4Bharat Svarah (~1.0 GB) -> datasets/Svarah/data/test-*.parquet
huggingface-cli download ai4bharat/Svarah --repo-type dataset --local-dir datasets/Svarah

# Telugu: IndicTTS Telugu (~6.0 GB) -> datasets/IndicTTS_Telugu/data/train-*.parquet
huggingface-cli download SPRINGLab/IndicTTS_Telugu --repo-type dataset --local-dir datasets/IndicTTS_Telugu
```

---

## 6. 5-Hour Dataset Preparation and Subsampling

Extract raw audio and generate the 5-hour training and 1-hour validation splits in accordance with assignment data constraints:

```bash
# English: 4.98 h train (3,419 utt.), 1.09 h val (746 utt.) across 104 speakers
python English/preprocess/create_svarah_splits.py

# Telugu: 4.9988 h aligned train (2,248 utt.), 1.0030 h val (445 utt.), 50/50 gender balance
python Telugu/preprocess/create_telugu_splits.py
```

Curated partitions are stored in:
- English: `datasets/prepared/english_speaker_independent/`
- Telugu: `datasets/prepared/telugu_prepared_v2/`

---

## 7. Audio Standardization, Forced Alignment & Acoustic Preprocessing

Execute audio downsampling, forced phonemic/akshara alignment, and acoustic feature extraction:

### 7.1. Audio Resampling (Telugu)
Downsample male studio recordings from 48 kHz to 16 kHz using a 64-tap Kaiser sinc filter:
```bash
python Telugu/preprocess/resample_telugu_audio.py
# Outputs 16 kHz WAVs to: preprocessing/Telugu/wavs_16k/
```

### 7.2. Forced Alignment
```bash
# English MFA alignment (outputs TextGrids to preprocessing/English/TextGrid/):
python English/preprocess.py --config English/config.yaml --stage align --jobs 8

# Telugu MMS CTC alignment (outputs token alignments to preprocessing/Telugu/alignments/):
python Telugu/preprocess/run_telugu_mms_alignment.py
```

### 7.3. Acoustic Feature Extraction
Extract 80-channel log mel-spectrograms, continuous $F_0$ (PyWorld Harvest), and frame energy ($L_2$-norm):
```bash
# English feature extraction (mel, pitch, energy, duration -> preprocessed_data/):
python English/preprocess.py --config English/config.yaml --stage convert

# Telugu feature extraction (mel, pitch, energy, duration -> preprocessed_data/):
python Telugu/preprocess/preprocess.py --config Telugu/config.yaml
```
Extracted features and normalization statistics are stored in `preprocessed_data/stats.json`, `preprocessed_data/mel/`, `preprocessed_data/pitch/`, `preprocessed_data/energy/`, and `preprocessed_data/duration/`.

---

## 8. Training Instructions (Kaggle Dual-GPU & Local)

Both models were trained for **100,000 steps** on dual NVIDIA Tesla T4 GPUs using PyTorch DataParallel with Automated Mixed Precision (AMP FP16) and a total batch size of 16 (8 utterances per GPU).

### 8.1. Kaggle Dual-GPU Execution
Self-contained Kaggle notebooks are provided in the repository:
* English: [`English/English_FastSpeech2_Kaggle.ipynb`](English/English_FastSpeech2_Kaggle.ipynb)
* Telugu: [`Telugu/Telugu_FastSpeech2_Kaggle.ipynb`](Telugu/Telugu_FastSpeech2_Kaggle.ipynb)

**Workflow**: Set the Kaggle accelerator to **GPU T4 x2**. The notebook automatically restores checkpoints across 12-hour sessions and reaches 100,000 steps without loss disruption.

### 8.2. Standalone Local / Server Training
Alternatively, run training directly on any local or server GPU:
```bash
# Train English FastSpeech 2 (100k steps)
python English/train.py --config English/config.yaml

# Train Telugu FastSpeech 2 (100k steps)
python Telugu/train.py --config Telugu/config.yaml
```

---

## 9. Inference & Speech Synthesis (Benchmark & Custom)

### 9.1. Synthesizing All Official Assignment Benchmark Sentences
To synthesize all ten prescribed benchmark test sentences (E1–E4 and T1–T4) and save them to `Results/<Language>/test_generated_audios/`:

```bash
# English benchmark suite (E1, E2a, E2b, E3, E4)
python English/inference.py --config English/config.yaml --test_all

# Telugu benchmark suite (T1, T2a, T2b, T3, T4)
python Telugu/inference.py --config Telugu/config.yaml --test_all
```

### 9.2. Synthesizing Custom Sentences (CLI)
Synthesize arbitrary raw text directly from the terminal with duration calibration ($d_{\text{control}} = 1.25$):

#### English Custom Text Synthesis:
```bash
# Custom English synthesis with calibrated duration control and reference speaker 27
python English/inference.py --config English/config.yaml \
    --text "Good morning to all, how are you?" \
    --duration_control 1.25 --speaker_id 27 --output_wav sample_en.wav
```

#### Telugu Custom Text Synthesis:
```bash
# Custom Telugu synthesis from text string or file
python Telugu/inference.py --config Telugu/config.yaml \
    --text "నమస్కారం, మీరు ఎలా ఉన్నారు?" \
    --duration_control 1.25 --speaker_id 0 --output_wav sample_te.wav
```

### 9.3. Speaker Selection & Multi-Speaker Architecture
The English FastSpeech 2 model is a multi-speaker architecture trained with a 256-dimensional learned speaker embedding table covering **104 distinct speakers** (IDs 0–103 mapped in [`English/speakers.json`](English/speakers.json)).

* **Default Speaker: Speaker ID 27 (`spk_029`)**
  * **Selection Rationale:** Selected as the primary reference speaker due to having the largest individual training utterance representation (118 utterances) in the AI4Bharat Svarah corpus, ensuring stable phonetic alignments and natural prosodic rendering.
  * **Vocal Profile:** Clear, natural Indian English female voice ($F_0 \approx 210\text{ Hz}$) with high intelligibility and smooth pitch trajectories.
  * **CLI Usage:** Pass `--speaker_id 27` (or any ID from 0 to 103) to select the target voice:
    ```bash
    python English/inference.py --config English/config.yaml --text "Testing multi-speaker synthesis." --speaker_id 27 --output_wav out27.wav
    ```

---

## 10. Benchmark Evaluation & Linguistic Results

All 10 benchmark sentences prescribed in **Section 3** of the assignment were synthesized using the duration-calibrated acoustic models and HiFi-GAN vocoder:

| ID | Lang | Benchmark Sentence / Linguistic Challenge | Duration | Generated Audio Path |
| :---: | :---: | :--- | :---: | :--- |
| **E1** | Eng | *The antidisestablishmentarianism movement gained considerable momentum during the parliamentary proceedings.* (Extreme polysyllabic word) | 5.09 s | [`Results/English/test_generated_audios/E1.wav`](Results/English/test_generated_audios/E1.wav) |
| **E2a** | Eng | *The orchestra's conduct during the ceremony was praised by the audience.* (Heteronym: noun stress on 1st syllable) | 3.37 s | [`Results/English/test_generated_audios/E2_a.wav`](Results/English/test_generated_audios/E2_a.wav) |
| **E2b** | Eng | *The researchers will conduct an experiment to evaluate the proposed method.* (Heteronym: verb stress on 2nd syllable) | 3.56 s | [`Results/English/test_generated_audios/E2_b.wav`](Results/English/test_generated_audios/E2_b.wav) |
| **E3** | Eng | *The rhyth  m of the orchestra changed unexpectedly.* (Typographical space normalized → *rhythm*) | 2.17 s | [`Results/English/test_generated_audios/E3.wav`](Results/English/test_generated_audios/E3.wav) |
| **E3 (verbatim)** | Eng | *The rhyth  m of the orchestra changed unexpectedly.* (Verbatim split with 120 ms pause token `@sp`) | 2.43 s | [`Results/English/test_generated_audios/E3_split_m.wav`](Results/English/test_generated_audios/E3_split_m.wav) |
| **E4** | Eng | *The meeting will start at 10:45 AM on 12 November 2026, so andaroo should arrive on time.* (Numbers, times & Telugu loanword *andaroo*) | 5.46 s | [`Results/English/test_generated_audios/E4.wav`](Results/English/test_generated_audios/E4.wav) |
| **T1** | Tel | న్యూరోఫార్మకోలాజికల్ పరిశోధనలో నూతన ఔషధాల ప్రభావాన్ని విశ్లేషించారు. (Heavy consonant conjuncts: *nyū*, *rma*, *l*) | 4.97 s | [`Results/Telugu/test_generated_audios/T1.wav`](Results/Telugu/test_generated_audios/T1.wav) |
| **T2a** | Tel | తాడుపై అతని పట్టు చాలా బలంగా ఉంది. (Semantic focus on *pattu* = grip) | 2.45 s | [`Results/Telugu/test_generated_audios/T2_a.wav`](Results/Telugu/test_generated_audios/T2_a.wav) |
| **T2b** | Tel | ఆమె పట్టు చీరను ధరించి వేడుకకు వచ్చింది. (Unstressed nominal modifier *pattu* = silk) | 2.86 s | [`Results/Telugu/test_generated_audios/T2_b.wav`](Results/Telugu/test_generated_audios/T2_b.wav) |
| **T3** | Tel | పులి అడ  విలో నిశ్శబ్దంగా నడిచింది. (Split word *ada vilo* robustness) | 2.58 s | [`Results/Telugu/test_generated_audios/T3.wav`](Results/Telugu/test_generated_audios/T3.wav) |
| **T4** | Tel | ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు ప్రారంభమవుతుంది, సమయానికి సిద్ధంగా ఉంచండి. (Complex date, numerals & clause boundaries) | 7.22 s | [`Results/Telugu/test_generated_audios/T4.wav`](Results/Telugu/test_generated_audios/T4.wav) |

---

## 11. Loss Curves & Convergence Analysis

| Language | Checkpoint Description | Step | Train Loss | Validation Loss | Selection Justification |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **English** | Numerical Lowest Val Loss | 11,000 | 2.5887 | **2.5505** | Muffled formants due to L1 blurriness trap (regression-to-mean). |
| **English** | **Selected Synthesis Model** | **61,000** | **0.8415** | **2.5855** | **Optimal Checkpoint**: Crisp consonants, sharp formants, superior audio fidelity. |
| **Telugu** | Numerical Lowest Val Loss | 30,000 | $\approx 0.8810^\dagger$ | **2.1309** | Early stage: lacks high-frequency consonant definition. |
| **Telugu** | **Selected Synthesis Model** | **74,000** | **0.6412** | **2.1457** | **Optimal Checkpoint**: Natural geminate closures, clear conjunct articulation. |

### English Loss Curves (100k Steps)
| English Training Loss | English Validation Loss |
| :---: | :---: |
| ![English Training Loss](Results/English/training_loss.png) | ![English Validation Loss](Results/English/validation_loss.png) |

### Telugu Loss Curves (100k Steps)
| Telugu Training Loss | Telugu Validation Loss |
| :---: | :---: |
| ![Telugu Training Loss](Results/Telugu/training_loss.png) | ![Telugu Validation Loss](Results/Telugu/validation_loss.png) |
