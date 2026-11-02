# Telugu FastSpeech 2 Model Checkpoints

Due to GitHub's standard file size limits, large pre-trained model weights (`*.pth`) and HiFi-GAN vocoder weights (`*.pth.tar`) are hosted externally on Google Drive.

---

## Google Drive Download Link

Download the Telugu model weights from the shared Google Drive folder:
👉 **[Google Drive Shared Model Folder](https://drive.google.com/drive/folders/1MYS5zcsyehxDN5n4XaKj5jCidpUZ-efw?usp=sharing)**

Navigate to the `Telugu/` subfolder in Google Drive:
- `latest_checkpoint.pth` (~400 MB)
- `best_model.pth` (~400 MB)
- `generator_universal.pth.tar` (~55 MB)

---

## File Placement Instructions

After downloading, place the files into this repository as follows:

| Downloaded File | Target Repository Path | Description |
| :--- | :--- | :--- |
| `latest_checkpoint.pth` | `Telugu/checkpoints/latest_checkpoint.pth` | Resumable training checkpoint (Step 56,000) containing model weights, Adam optimizer states, and learning rate scheduler state. |
| `best_model.pth` | `Telugu/checkpoints/best_model.pth` | Top-performing validation model (Step 30,000, Val Loss: 2.1309) for high-fidelity speech synthesis. |
| `generator_universal.pth.tar` | `Telugu/inference/hifigan/generator_universal.pth.tar` | HiFi-GAN 16 kHz universal neural vocoder weights (also copy to `Telugu/train/hifigan/`). |

---

## Checkpoint Details

- **`latest_checkpoint.pth` (Step 56,000 | Epoch 400 | Val Loss: 2.1509)**:
  Used to **resume training** up to Step 100,000 seamlessly without starting from scratch.
  ```bash
  python Telugu/train/train.py --config Telugu/config.yaml --resume
  ```

- **`best_model.pth` (Step 30,000 | Epoch 214 | Val Loss: 2.1309)**:
  Used for **inference and evaluation** (generates natural, accurate Telugu speech).
  ```bash
  python Telugu/inference/inference.py --config Telugu/config.yaml --checkpoint Telugu/checkpoints/best_model.pth --speaker_id 0 --phonemes "న ్ య ూ ర ో ఫ ా ర ్ మ క ో ల ా జ ి క ల ్" --output_wav sample.wav
  ```

