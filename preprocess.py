#!/bin/python3
import os
import numpy as np
import librosa
import cv2
import pywt
from tqdm import tqdm

# ---------------------------------------------------------
# Basic audio loading and trimming
# ---------------------------------------------------------
def load_and_trim(path, sr=22050, duration=2.0):
    y, _ = librosa.load(path, sr=sr)
    max_len = int(sr * duration)
    if len(y) < max_len:
        y = np.pad(y, (0, max_len - len(y)))
    else:
        y = y[:max_len]
    return y

# ---------------------------------------------------------
# Mel spectrogram computation
# ---------------------------------------------------------
def compute_mel(y, sr=22050, n_fft=2048, hop_length=512, n_mels=128):
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft,
        hop_length=hop_length, n_mels=n_mels
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db.astype(np.float32)

# ---------------------------------------------------------
# Wavelet scalogram computation
# ---------------------------------------------------------
def compute_scalogram(y, sr=22050, wavelet='morl', scales=np.arange(1, 129)):
    coeffs, _ = pywt.cwt(y, scales, wavelet, sampling_period=1/sr)
    scalogram = np.abs(coeffs)
    scalogram_db = 20 * np.log10(scalogram + 1e-6)
    return scalogram_db.astype(np.float32)

# ---------------------------------------------------------
# Normalization
# ---------------------------------------------------------
def normalize(feat):
    feat -= feat.min()
    feat /= (feat.max() + 1e-6)
    return feat

# ---------------------------------------------------------
# Main conversion pipeline
# ---------------------------------------------------------
def wav_to_features(path, mode='mel', resize=(128, 128), **kwargs):
    sr = kwargs.get('sr', 22050)
    duration = kwargs.get('duration', 2.0)

    y = load_and_trim(path, sr, duration)

    if mode == 'mel':
        feat = compute_mel(y, sr)
    elif mode == 'wavelet':
        feat = compute_scalogram(y, sr)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    feat = normalize(feat)

    if resize:
        feat = cv2.resize(feat, resize, interpolation=cv2.INTER_AREA)

    return feat.astype(np.float32)

# ---------------------------------------------------------
# Directory walker
# ---------------------------------------------------------
def preprocess_dataset(input_dir,
                       output_dir,
                       mode="wavelet",
                       resize=(128, 128)):
    os.makedirs(output_dir, exist_ok=True)

    for sub in ["tb_positive", "tb_negative"]:
        src_dir = os.path.join(input_dir, sub)
        if not os.path.isdir(src_dir):
            print(f"Skipping missing directory: {src_dir}")
            continue

        for root, _, files in os.walk(src_dir):
            for f in tqdm(files, desc=f"Processing {sub}", ncols=80):
                if not f.lower().endswith(".wav"):
                    continue

                in_path = os.path.join(root, f)
                rel_path = os.path.relpath(in_path, input_dir)
                npy_path = os.path.join(output_dir,
                                        os.path.splitext(rel_path)[0] + ".npy")

                os.makedirs(os.path.dirname(npy_path), exist_ok=True)

                if os.path.exists(npy_path):
                    continue  # skip already processed files

                feat = wav_to_features(in_path, mode=mode, resize=resize)
                np.save(npy_path, feat)

    print(f"\nPreprocessing complete.\nSaved features to: {output_dir}\n")

# ---------------------------------------------------------
# Main entry point with switch-case logic
# ---------------------------------------------------------
if __name__ == "__main__":
    print("Select preprocessing mode:")
    print("1. Train data -> preprocessed_train_data/")
    print("2. Test data  -> preprocessed_test_data/")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        preprocess_dataset(
            input_dir="train_data",
            output_dir="preprocessed_train_data",
            mode="wavelet",
            resize=(128, 128)
        )
    elif choice == "2":
        preprocess_dataset(
            input_dir="test_data",
            output_dir="preprocessed_test_data",
            mode="wavelet",
            resize=(128, 128)
        )
    else:
        print("Invalid choice. Exiting.")
