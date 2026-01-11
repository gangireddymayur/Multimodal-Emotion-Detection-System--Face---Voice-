import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
# SIMPLE MLP VOICE MODEL
# =========================
class VoiceEmotionModel(nn.Module):
    def __init__(self, n_mfcc=40, num_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_mfcc, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# Emotion labels
EMOTIONS = ["Angry", "Happy", "Sad", "Neutral"]

# Load model (demo mode – random weights but consistent)
model = VoiceEmotionModel()
model.eval()

# =========================
# PREDICTION FUNCTION
# =========================
def predict_voice_emotion(audio_path):
    # Load audio
    y, sr = librosa.load(audio_path, sr=16000)

    # Extract MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

    # Average over time → (40,)
    mfcc_mean = np.mean(mfcc, axis=1)

    # Convert to tensor → (1, 40)
    x = torch.tensor(mfcc_mean, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        idx = torch.argmax(probs, dim=1).item()

    return EMOTIONS[idx], probs[0][idx].item()
