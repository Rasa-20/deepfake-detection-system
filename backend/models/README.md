# Model: Inference & Training Artifacts

This folder contains:
- `training_notebook.ipynb` – Google Colab notebook used for training
- `inference.py` – script to run predictions on new videos (REAL vs FAKE)
- `utils.py` – preprocessing helpers for frames and audio (MFCCs)
- `weights/` – directory to store trained model weights (not included in repo)

---

## 1) Environment Setup
Install the dependencies (backend has the full list, but minimally you need):
```bash
pip install torch torchvision torchaudio opencv-python librosa moviepy numpy
