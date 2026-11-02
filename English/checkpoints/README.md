# English FastSpeech 2 Model Checkpoints

Due to GitHub's standard file size limits, large pre-trained model weights (`*.pth`) and HiFi-GAN vocoder weights (`*.pth.tar`) are hosted externally on Google Drive.

---

## Google Drive Download Link

Download the English model weights from the shared Google Drive folder:
👉 **[Google Drive Shared Model Folder](https://drive.google.com/drive/folders/1MYS5zcsyehxDN5n4XaKj5jCidpUZ-efw?usp=sharing)**

Navigate to the `English/` subfolder in Google Drive:
- `latest_checkpoint.pth` (~399 MB)
- `best_model.pth` (~399 MB)
- `generator_universal.pth.tar` (~55 MB)

---

## File Placement Instructions

After downloading, place the files into this repository as follows:

| Downloaded File | Target Repository Path | Description |
| :--- | :--- | :--- |
| `latest_checkpoint.pth` | `English/checkpoints/latest_checkpoint.pth` | Resumable training checkpoint (Step 53,000) containing model weights, Adam optimizer states, and learning rate scheduler state. |
| `best_model.pth` | `English/checkpoints/best_model.pth` | Top-performing validation model (Step 11,000, Val Loss: 2.5505) for high-fidelity speech synthesis. |
| `generator_universal.pth.tar` | `English/inference/hifigan/generator_universal.pth.tar` | HiFi-GAN 16 kHz universal neural vocoder weights (also copy to `English/train/hifigan/`). |

---

## Checkpoint Details

- **`latest_checkpoint.pth` (Step 53,000 | Epoch 249 | Val Loss: 2.5926)**:
  Used to **resume training** up to Step 100,000 seamlessly without starting from scratch.
  ```bash
  python English/train/train.py --config English/config.yaml --resume
  ```

- **`best_model.pth` (Step 11,000 | Epoch 52 | Val Loss: 2.5505)**:
  Used for **inference and evaluation** (generates crisp audio without over-smoothing).
  ```bash
  python English/inference/inference.py --config English/config.yaml --checkpoint English/checkpoints/best_model.pth --speaker_id 28 --phonemes "T h e   a n t i d i s e s t a b l i s h m e n t a r i a n i s m   m o v e m e n t" --output_wav sample.wav
  ```

