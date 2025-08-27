# model/utils.py
import os
import random
import numpy as np
import cv2
import librosa
from moviepy.editor import VideoFileClip

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def pad_or_trim_time_axis(x: np.ndarray, target: int, axis: int = 0):
    """
    Pads with zeros or trims along the given axis to match target length.
    """
    cur = x.shape[axis]
    if cur == target:
        return x
    if cur > target:
        slices = [slice(None)] * x.ndim
        slices[axis] = slice(0, target)
        return x[tuple(slices)]
    # pad
    pad_width = [(0,0)] * x.ndim
    pad_width[axis] = (0, target - cur)
    return np.pad(x, pad_width, mode="constant")

def preprocess_video(video_path: str, fixed_num_frames: int = 128, resize_hw=(112,112)):
    """
    Returns np.ndarray of shape (T, H, W, C) in [0,1], RGB, with T=fixed_num_frames.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, resize_hw, interpolation=cv2.INTER_AREA)
        frames.append(frame)
    cap.release()

    if len(frames) == 0:
        raise RuntimeError("No frames extracted. Check the video path/codec.")

    arr = np.stack(frames).astype(np.float32) / 255.0  # (T, H, W, C)
    arr = pad_or_trim_time_axis(arr, fixed_num_frames, axis=0)

    # Optional: uniform sampling instead of plain trim/pad
    if arr.shape[0] > fixed_num_frames:
        idxs = np.linspace(0, arr.shape[0]-1, fixed_num_frames).astype(int)
        arr = arr[idxs]
    return arr

def extract_audio_array(video_path: str, target_sr: int = 16000):
    """
    Extracts audio samples from video using moviepy (returns mono float32, sr).
    """
    with VideoFileClip(video_path) as clip:
        audio = clip.audio
        if audio is None:
            raise RuntimeError("No audio track found in the video.")
        # to_soundarray returns float32 in [-1,1], shape (samples, channels)
        arr = audio.to_soundarray(fps=target_sr).astype(np.float32)
    if arr.ndim == 2:
        arr = arr.mean(axis=1)  # mono
    return arr, target_sr

def preprocess_audio(video_path: str, n_mfcc: int = 40, fixed_steps: int = 200):
    """
    Returns MFCCs as np.ndarray of shape (S, n_mfcc), S=fixed_steps
    """
    y, sr = extract_audio_array(video_path, target_sr=16000)
    # Librosa MFCC: (n_mfcc, steps)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc = mfcc.T  # (steps, n_mfcc)
    mfcc = pad_or_trim_time_axis(mfcc, fixed_steps, axis=0)
    return mfcc.astype(np.float32)
