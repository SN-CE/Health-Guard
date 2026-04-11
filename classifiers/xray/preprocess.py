#!/bin/python3
# preprocess.py - X-ray preprocessing for TB detection
import os
from PIL import Image
from tqdm import tqdm


def preprocess_dataset(input_dir, output_dir, resize=(224, 224)):
    os.makedirs(output_dir, exist_ok=True)

    for sub in ['tb_positive', 'tb_negative']:
        src_dir = os.path.join(input_dir, sub)
        if not os.path.isdir(src_dir):
            print(f"Skipping missing directory: {src_dir}")
            continue

        out_sub_dir = os.path.join(output_dir, sub)
        os.makedirs(out_sub_dir, exist_ok=True)

        files = [f for f in os.listdir(src_dir)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for f in tqdm(files, desc=f"Processing {sub}", ncols=80):
            in_path = os.path.join(src_dir, f)
            out_path = os.path.join(out_sub_dir, os.path.splitext(f)[0] + '.png')

            if os.path.exists(out_path):
                continue

            try:
                img = Image.open(in_path).convert('RGB')
                img = img.resize(resize, Image.LANCZOS)
                img.save(out_path)
            except Exception as e:
                print(f"[WARN] Skipping {in_path}: {e}")

    print(f"\nDone. Saved to: {output_dir}")


if __name__ == '__main__':
    preprocess_dataset(
        input_dir='../../data/raw/xray',
        output_dir='../../data/preprocessed/xray',
        resize=(224, 224)
    )
