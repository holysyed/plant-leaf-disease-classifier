"""Simple browser demo for the trained plant disease classifier."""

from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


MODEL_DIR = Path("artifacts")
MODEL_PATH = MODEL_DIR / "plant_disease_model.keras"
LABELS_PATH = MODEL_DIR / "class_names.json"

st.set_page_config(page_title="Leaf Health Classifier", page_icon="🌿")
st.title("🌿 Leaf Health Classifier")
st.write("Upload a leaf photo to see which trained class the model predicts.")


@st.cache_resource
def load_classifier():
    import json

    model = tf.keras.models.load_model(MODEL_PATH)
    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    return model, labels


if not MODEL_PATH.is_file() or not LABELS_PATH.is_file():
    st.info("Train the model first: `python train.py`")
else:
    uploaded = st.file_uploader("Choose a leaf image", type=("jpg", "jpeg", "png"))
    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded leaf", use_container_width=True)
        model, labels = load_classifier()
        image_shape = model.input_shape[1:3]
        resized = image.resize((image_shape[1], image_shape[0]))
        pixels = np.asarray(resized, dtype=np.float32)[None, ...]
        batch = tf.keras.applications.mobilenet_v2.preprocess_input(pixels)
        probabilities = model.predict(batch, verbose=0)[0]
        ranking = np.argsort(probabilities)[::-1][: min(3, len(labels))]
        winner = int(ranking[0])
        st.subheader(labels[winner])
        st.metric("Model confidence", f"{probabilities[winner]:.1%}")
        st.caption("This is a model estimate, not an agronomic diagnosis.")
        st.write("Top predictions")
        for index in ranking:
            st.write(f"{labels[index]} — {probabilities[index]:.1%}")
