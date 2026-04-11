#!/bin/python3
# infer.py - Inference for TB detection
import os
import sys
import argparse
import torch
import numpy as np
import librosa
import cv2
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'classifiers/audio'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'classifiers/xray'))

from classifiers.audio.model import AudioClassifier
from classifiers.xray.model  import XrayClassifier


# ===== PREPROCESSING =====

def preprocess_audio(path, sr=22050, duration=2.0, resize=(128, 128)):
    y, _ = librosa.load(path, sr=sr)
    max_len = int(sr * duration)
    if len(y) < max_len:
        y = np.pad(y, (0, max_len - len(y)))
    else:
        y = y[:max_len]

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048,
                                          hop_length=512, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max).astype(np.float32)

    mel_db -= mel_db.min()
    mel_db /= (mel_db.max() + 1e-6)

    mel_db = cv2.resize(mel_db, resize, interpolation=cv2.INTER_AREA)

    # (1, 1, H, W) — batch size 1, single channel
    return torch.from_numpy(mel_db).unsqueeze(0).unsqueeze(0).float()


def preprocess_xray(path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(path).convert('RGB')
    return transform(img).unsqueeze(0).float()  # (1, 3, 224, 224)


# ===== INFERENCE =====

def run(audio_path=None, xray_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if audio_path is None and xray_path is None:
        print("Error: at least one input must be provided.")
        sys.exit(1)

    scores = []

    if audio_path is not None:
        if not os.path.exists('weights/audio_classifier.pt'):
            print("Error: weights/audio_classifier.pt not found.")
            sys.exit(1)

        audio_model = AudioClassifier().to(device)
        audio_model.load_state_dict(torch.load('weights/audio_classifier.pt',
                                                map_location=device))
        audio_model.eval()

        with torch.no_grad():
            x = preprocess_audio(audio_path).to(device)
            score = torch.sigmoid(audio_model.classify(x)).item()
            scores.append(('Audio', score))

    if xray_path is not None:
        if not os.path.exists('weights/xray_classifier.pt'):
            print("Error: weights/xray_classifier.pt not found.")
            sys.exit(1)

        xray_model = XrayClassifier().to(device)
        xray_model.load_state_dict(torch.load('weights/xray_classifier.pt',
                                               map_location=device))
        xray_model.eval()

        with torch.no_grad():
            x = preprocess_xray(xray_path).to(device)
            score = torch.sigmoid(xray_model.classify(x)).item()
            scores.append(('X-ray', score))

    # ===== OUTPUT =====
    print("\n" + "=" * 40)
    for name, score in scores:
        print(f"{name} score:   {score:.4f}")

    final_score = np.mean([s for _, s in scores])
    diagnosis   = "TB Positive" if final_score > 0.5 else "TB Negative"

    if len(scores) > 1:
        print(f"Final score:  {final_score:.4f}")

    print(f"Diagnosis:    {diagnosis}")
    print("=" * 40)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Health-Guard TB inference')
    parser.add_argument('--audio', type=str, default=None, help='Path to .wav cough recording')
    parser.add_argument('--xray',  type=str, default=None, help='Path to chest X-ray image')
    args = parser.parse_args()

    run(audio_path=args.audio, xray_path=args.xray)
