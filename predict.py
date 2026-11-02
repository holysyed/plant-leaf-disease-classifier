"""Predict the class of a single plant leaf image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Path to a leaf image")
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()

    model_path = args.model_dir / "plant_disease_model.keras"
    labels_path = args.model_dir / "class_names.json"
    if not model_path.is_file() or not labels_path.is_file():
        raise SystemExit("Model files not found. Train the model first with python train.py.")
    if not args.image.is_file():
        raise SystemExit(f"Image not found: {args.image}")

    model = tf.keras.models.load_model(model_path)
    image_shape = model.input_shape[1:3]
    with Image.open(args.image) as source:
        image = source.convert("RGB").resize((image_shape[1], image_shape[0]))
        pixels = np.asarray(image, dtype=np.float32)
    batch = tf.keras.applications.mobilenet_v2.preprocess_input(pixels[None, ...])

    class_names = json.loads(labels_path.read_text(encoding="utf-8"))
    probabilities = model.predict(batch, verbose=0)[0]
    best_index = int(np.argmax(probabilities))
    print(f"Prediction: {class_names[best_index]}")
    print(f"Model confidence: {probabilities[best_index]:.1%}")
    print("This is a model estimate, not an agronomic diagnosis.")


if __name__ == "__main__":
    try:
        main()
    except UnidentifiedImageError as error:
        raise SystemExit(f"Could not read that image: {error}") from error
