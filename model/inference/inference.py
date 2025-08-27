# model/inference.py
# Run:  python inference.py --video path/to/video.mp4 --weights model/weights/model.pth
# Requirements: torch, torchvision, torchaudio (optional), opencv-python, librosa, numpy, moviepy

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from pathlib import Path

from utils import preprocess_video, preprocess_audio, set_seed

LABELS = ["REAL", "FAKE"]  # 0 -> REAL, 1 -> FAKE

class AVFusionModel(nn.Module):
    """
    Minimal inference-time model:
    - Video encoder: R(2+1)D-18 -> 512-d embedding
    - Audio encoder: BiLSTM over MFCCs -> 256-d embedding
    - Late fusion: concat(vid,aud) -> FC -> logits(2)
    NOTE: This is a matching skeleton to load your trained weights.
    """
    def __init__(self, vid_embed_dim=512, aud_hidden=128, aud_layers=1, num_classes=2):
        super().__init__()

        # ---- Video encoder (R(2+1)D-18) ----
        base = torchvision.models.video.r2plus1d_18(pretrained=False)
        # strip classifier; keep feature extractor
        base.fc = nn.Identity()
        self.video_encoder = base  # outputs 512-d

        # ---- Audio encoder (BiLSTM over MFCCs) ----
        self.audio_lstm = nn.LSTM(
            input_size=40,        # n_mfcc
            hidden_size=aud_hidden,
            num_layers=aud_layers,
            batch_first=True,
            bidirectional=True
        )
        self.audio_proj = nn.Sequential(
            nn.Linear(aud_hidden * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )

        # ---- Late fusion + classifier ----
        self.classifier = nn.Sequential(
            nn.Linear(512 + 256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, vid_tensor, mfcc_tensor):
        """
        vid_tensor:  (B, C=3, T=frames, H, W) float in [0,1]
        mfcc_tensor: (B, S=steps, 40)
        """
        # Video path
        v = self.video_encoder(vid_tensor)   # (B, 512)

        # Audio path
        a_out, _ = self.audio_lstm(mfcc_tensor)  # (B, S, 2*hidden)
        # take last timestep (or mean-pool)
        a = a_out.mean(dim=1)                    # (B, 2*hidden)
        a = self.audio_proj(a)                   # (B, 256)

        # Late fusion
        z = torch.cat([v, a], dim=1)             # (B, 768)
        logits = self.classifier(z)              # (B, 2)
        return logits


def load_weights(model: nn.Module, weights_path: Path, map_location="cpu"):
    ckpt = torch.load(str(weights_path), map_location=map_location)
    # Accept either {'state_dict': ...} or a raw state_dict
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to input video (.mp4)")
    parser.add_argument("--weights", default="model/weights/model.pth", help="Path to trained weights")
    parser.add_argument("--frames", type=int, default=128, help="Fixed number of frames")
    parser.add_argument("--audio_steps", type=int, default=200, help="Fixed MFCC steps")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Build model and load weights
    model = AVFusionModel()
    weights_path = Path(args.weights)
    if weights_path.exists():
        load_weights(model, weights_path, map_location=device)
    else:
        print(f"[WARN] Weights not found at {weights_path}. Running with random weights (for structure check only).")

    model.to(device).eval()

    # --- Preprocess inputs ---
    vid = preprocess_video(args.video, fixed_num_frames=args.frames, resize_hw=(112, 112))  # (T, H, W, C)
    mfcc = preprocess_audio(args.video, n_mfcc=40, fixed_steps=args.audio_steps)            # (S, 40)

    # convert to tensors with expected shapes
    vid_tensor = torch.from_numpy(vid).permute(3, 0, 1, 2).unsqueeze(0).float().to(device)  # (1,3,T,H,W)
    mfcc_tensor = torch.from_numpy(mfcc).unsqueeze(0).float().to(device)                    # (1,S,40)

    with torch.no_grad():
        logits = model(vid_tensor, mfcc_tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        pred_idx = int(probs.argmax())
        label = LABELS[pred_idx]
        confidence = float(probs[pred_idx])

    print(f"Prediction: {label}  (confidence: {confidence:.4f})")

if __name__ == "__main__":
    main()
