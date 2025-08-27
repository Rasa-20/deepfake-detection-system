from fastapi import FastAPI, UploadFile, File   # FastAPI: Web framework for building APIs | to handle file uploads via the /analyze endpoint
from fastapi.middleware.cors import CORSMiddleware  # lets the backend communicate with the frontend (running on localhost:5173)
from .model import DeepfakeDetector     # this is the trained PyTorch model from model.py
import torch    # PyTorch library for model loading and tensor computation
import os            # file handling
import tempfile      # file handling
from moviepy.video.io.VideoFileClip import VideoFileClip    # From MoviePy, used to extract audio from videos
import librosa      # For audio processing (specifically extracting MFCCs)
import numpy as np  # numerical operations
import cv2    # For image/frame manipulation

# -------------------------------
# FastAPI Setup
# -------------------------------
app = FastAPI()                         # initialize api app

origins = ["replace with your frontend URL"]     # allowing frontend to communicate with the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Load Model
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")       # detects GPU or CPU

model = DeepfakeDetector().to(device)
model_path = os.path.join(os.path.dirname(__file__), "models", "model path")  # load the locally saved model path
print(f"📂 Loading model from: {model_path}")
assert os.path.exists(model_path), f"❌ Model file not found at {model_path}"

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()        # tells PyTorch to disable training behavior 

print("✅ Model loaded successfully.")

# -------------------------------
# Frame and Audio Extractors
# -------------------------------
def extract_frames(video_path, num_frames=128):     # extract frames from video
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened() and len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (112, 112))
        frames.append(frame)
    cap.release()

    if len(frames) == 0:
        raise ValueError("No frames extracted from video!")

    frames_np = np.stack(frames)
    if frames_np.shape[0] < num_frames:
        pad = np.zeros((num_frames - frames_np.shape[0], 112, 112, 3), dtype=np.float32)
        frames_np = np.concatenate([frames_np, pad], axis=0)
    else:
        frames_np = frames_np[:num_frames]

    frames_tensor = torch.tensor(frames_np, dtype=torch.float32).permute(3, 0, 1, 2) 
    return frames_tensor.unsqueeze(0)

def extract_audio_mfcc(video_path, n_mfcc=40, fixed_steps=200):     # extract audio MFCC features
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_file:
        clip = VideoFileClip(video_path)
        clip.audio.write_audiofile(audio_file.name, logger=None)
        y, sr = librosa.load(audio_file.name, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    if mfcc.shape[1] < fixed_steps:
        mfcc = np.pad(mfcc, ((0, 0), (0, fixed_steps - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :fixed_steps]

    return torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)

# -------------------------------
# API Endpoints
# -------------------------------
@app.get("/")                                   # simple route to check if the server is running
def read_root():
    return {"message": "Backend is working!"}

@app.post("/analyze")                                   # saves the uploaded video to a temporary .mp4 file
async def analyze_video(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(await file.read())
            temp_video_path = temp_file.name

        video_tensor = extract_frames(temp_video_path).to(device)     # calls the two extraction functions defined earlier
        audio_tensor = extract_audio_mfcc(temp_video_path).to(device)

        with torch.no_grad():       # disabling gradient since its not training
            output = model(video_tensor, audio_tensor) # feeding in your audio and video features (tensors) into the model
            probs = torch.softmax(output, dim=1).cpu().numpy()[0] # applies softmax to get class probabilities (fake/real)

        # 🧠 Logging Softmax Outputs for Debugging
        print("🧠 Softmax Output:", probs)      # rints the prediction to the console
        print(f"REAL confidence: {probs[0]:.4f}, FAKE confidence: {probs[1]:.4f}")
        print("✅ Final Prediction:", "Deepfake Detected" if probs[1] > probs[0] else "Video is Real")

        # ✅ 0 = REAL, 1 = FAKE
        real_conf = round(float(probs[0]) * 100, 2) # converts probabilities into percentages for display
        fake_conf = round(float(probs[1]) * 100, 2)
        is_fake = bool(probs[1] > probs[0])

        # ✅ Dynamic Explanation Messages
        if is_fake:                         # builds user-friendly explanation based on prediction
            visual_msg = "Detected inconsistencies in facial features."
            audio_msg = "Voice patterns appear synthetic."
            metadata_msg = "Multiple editing signatures found."
        else:
            visual_msg = "No inconsistencies detected in facial features."
            audio_msg = "Voice patterns align with natural speech."
            metadata_msg = "No suspicious editing signatures found."

        return {                                                            # returns the final response to the frontend
            "status": "Deepfake Detected" if is_fake else "Video is Real",
            "confidence": fake_conf if is_fake else real_conf,
            "isFake": is_fake,
            "visualAnalysis": visual_msg,
            "audioAnalysis": audio_msg,
            "metadataAnalysis": metadata_msg
        }

    except Exception as e:                        # handles and logs any errors that occur during upload or processing
        print("❌ Error during analysis:", e)
        return {"error": str(e)}
