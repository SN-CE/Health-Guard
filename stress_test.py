#!/bin/python3
# stress_test.py - Batch inference for TB detection
import os
import sys
import argparse
import torch
import numpy as np
import librosa
import cv2
from PIL import Image

from classifiers.audio.model  import AudioClassifier
from classifiers.xray.model   import XrayClassifier
from classifiers.xray.dataset import val_transform


class CustomFormatter(argparse.RawDescriptionHelpFormatter):
    def __init__(self, prog):
        super().__init__(prog, max_help_position=40)



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

    return torch.from_numpy(mel_db).unsqueeze(0).unsqueeze(0).float()


def preprocess_xray(path):
    img = Image.open(path).convert('RGB')
    return val_transform(img).unsqueeze(0).float()


# ===== BATCH INFERENCE =====

def run(audio_dir=None, xray_dir=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if audio_dir is None and xray_dir is None:
        print("Error: at least one directory must be provided.")
        sys.exit(1)

    # load models once
    audio_model = None
    xray_model  = None

    if audio_dir is not None:
        if not os.path.exists('weights/audio_classifier.pt'):
            print("Error: weights/audio_classifier.pt not found.")
            sys.exit(1)
        audio_model = AudioClassifier().to(device)
        audio_model.load_state_dict(torch.load('weights/audio_classifier.pt',
                                                map_location=device))
        audio_model.eval()
        print("Audio model loaded.")

    if xray_dir is not None:
        if not os.path.exists('weights/xray_classifier.pt'):
            print("Error: weights/xray_classifier.pt not found.")
            sys.exit(1)
        xray_model = XrayClassifier().to(device)
        xray_model.load_state_dict(torch.load('weights/xray_classifier.pt',
                                               map_location=device))
        xray_model.eval()
        print("X-ray model loaded.")

    # build stem → full path maps for each directory
    audio_files = {}
    xray_files  = {}

    if audio_dir is not None:
        for fn in os.listdir(audio_dir):
            if fn.lower().endswith('.wav'):
                stem = os.path.splitext(fn)[0]
                audio_files[stem] = os.path.join(audio_dir, fn)

    if xray_dir is not None:
        for fn in os.listdir(xray_dir):
            if fn.lower().endswith(('.png', '.jpg', '.jpeg')):
                stem = os.path.splitext(fn)[0]
                xray_files[stem] = os.path.join(xray_dir, fn)

    # union of all stems
    all_stems = sorted(set(audio_files) | set(xray_files))

    if not all_stems:
        print("No files found.")
        sys.exit(1)

    print(f"\nRunning inference on {len(all_stems)} samples...")
    print("=" * 60)

    for stem in all_stems:
        scores = []
        modes  = []

        if stem in audio_files and audio_model is not None:
            try:
                with torch.no_grad():
                    x = preprocess_audio(audio_files[stem]).to(device)
                    score = torch.sigmoid(audio_model.classify(x)).item()
                scores.append(score)
                modes.append(f"audio={score:.4f}")
            except Exception as e:
                modes.append(f"audio=ERROR({e})")

        if stem in xray_files and xray_model is not None:
            try:
                with torch.no_grad():
                    x = preprocess_xray(xray_files[stem]).to(device)
                    score = torch.sigmoid(xray_model.classify(x)).item()
                scores.append(score)
                modes.append(f"xray={score:.4f}")
            except Exception as e:
                modes.append(f"xray=ERROR({e})")

        if not scores:
            print(f"{stem:<40} ERROR: no valid scores")
            continue

        final_score = np.mean(scores)
        diagnosis   = "TB Positive" if final_score > 0.5 else "TB Negative"
        mode_str    = "  |  ".join(modes)

        print(f"{stem:<40} [{mode_str}]  →  final={final_score:.4f}  {diagnosis}")

    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Health-Guard batch inference stress test',
        epilog=(
            "examples:\n"
            "  python stress_test.py --audio-dir samples/audio/\n"
            "  python stress_test.py --xray-dir samples/xray/\n"
            "  python stress_test.py --audio-dir samples/audio/ --xray-dir samples/xray/"
        ),
        formatter_class=CustomFormatter
    )
    parser.add_argument('--audio-dir', type=str, default=None, help='Directory of .wav cough recordings')
    parser.add_argument('--xray-dir',  type=str, default=None, help='Directory of chest X-ray images')
    args = parser.parse_args()

    run(audio_dir=args.audio_dir, xray_dir=args.xray_dir)

