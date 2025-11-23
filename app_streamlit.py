import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageStat, ImageFilter
from tensorflow.keras.preprocessing import image
import os
import requests

# --------------------------------------------------------
# PAGE CONFIG + CSS
# --------------------------------------------------------
st.set_page_config(page_title="🍎 Fruit Freshness Detector", layout="centered")

page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: linear-gradient(145deg, #0f2027, #203a43, #2c5364);
    color: white;
}
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}
div.stButton > button:first-child {
    background-color: #f05454;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 16px;
    font-weight: bold;
}
div.stButton > button:hover {
    background-color: #ff7b54;
    color: black;
}
.uploadedImage {
    border-radius: 15px;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)


# --------------------------------------------------------
# ROBUST MODEL LOADER
# --------------------------------------------------------
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "fruit_freshness_model.h5")

# OPTIONS:
# "local"  -> model file is already in model/ folder
# "gdrive" -> download model from google drive using FILE_ID
# "url"    -> download from direct URL (S3, GCS, etc.)
MODEL_STRATEGY = "local"   # change before deploying if needed

# If using Google Drive:
MODEL_GDRIVE_ID = "YOUR_GOOGLE_DRIVE_FILE_ID"

# If using direct URL:
MODEL_URL = "https://your-bucket.s3.amazonaws.com/fruit_freshness_model.h5"


def download_from_url(url, dest_path):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(8192):
                if chunk:
                    f.write(chunk)


def download_from_gdrive(file_id, dest_path):
    try:
        import gdown
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        gdown.download(id=file_id, output=dest_path, quiet=False)
    except Exception:
        # Fallback manual Drive handling
        URL = "https://drive.google.com/uc?export=download&id=" + file_id
        session = requests.Session()
        response = session.get(URL, stream=True)
        token = None
        for k, v in response.cookies.items():
            if k.startswith("download_warning"):
                token = v

        if token:
            URL = URL + "&confirm=" + token
        download_from_url(URL, dest_path)


@st.cache_resource
def load_model():
    os.makedirs(MODEL_DIR, exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        # Download if not present
        if MODEL_STRATEGY == "local":
            raise FileNotFoundError(
                "MODEL_STRATEGY='local' but model is not in model/ folder."
            )

        elif MODEL_STRATEGY == "gdrive":
            if MODEL_GDRIVE_ID == "YOUR_GOOGLE_DRIVE_FILE_ID":
                raise ValueError("Please set a valid Google Drive FILE_ID.")
            download_from_gdrive(MODEL_GDRIVE_ID, MODEL_PATH)

        elif MODEL_STRATEGY == "url":
            if MODEL_URL.startswith("https://your-bucket"):
                raise ValueError("Please set a valid direct MODEL_URL.")
            download_from_url(MODEL_URL, MODEL_PATH)

        else:
            raise ValueError("Unknown MODEL_STRATEGY")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model file missing even after download attempt.")

    return tf.keras.models.load_model(MODEL_PATH)


# Show loading spinner
with st.spinner("Loading model... Please wait."):
    model = load_model()
st.success("Model Loaded Successfully!")

# ensure correct order from training
class_labels = ['freshapples', 'freshbanana', 'freshoranges',
                'rottenapples', 'rottenbanana', 'rottenoranges']


# --------------------------------------------------------
# PREDICTION FUNCTION
# --------------------------------------------------------
def predict_fruit(img):
    img_resized = img.resize((128, 128))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    predictions = model.predict(img_array)
    idx = np.argmax(predictions)
    fruit_class = class_labels[idx]
    confidence = np.max(predictions)

    return fruit_class, confidence


# --------------------------------------------------------
# ADVANCED ANALYZER
# --------------------------------------------------------
def analyze_image_advanced(img, fruit_class):
    img_rgb = img.convert("RGB").resize((128, 128))
    stat = ImageStat.Stat(img_rgb)
    brightness = sum(stat.mean) / 3
    vividness = np.std(np.array(img_rgb)) / 64

    img_np = np.array(img_rgb)
    brown_mask = (
        (img_np[:, :, 0] > 80)
        & (img_np[:, :, 1] < 90)
        & (img_np[:, :, 2] < 70)
    )
    brown_ratio = np.sum(brown_mask) / (128 * 128)

    edges = img_rgb.filter(ImageFilter.FIND_EDGES)
    roughness = np.mean(np.array(edges)) / 255

    freshness_score = (
        (brightness / 255) * 0.4
        + vividness * 0.3
        + (1 - brown_ratio) * 0.2
        + (1 - roughness) * 0.1
    )
    freshness_score = float(np.clip(freshness_score, 0, 1))

    # Fresh vs Rotten logic
    if "rotten" in fruit_class:
        if freshness_score > 0.75:
            age = "2–3 days old ⚠️"
            life = "Already showing signs of rot — discard soon."
            health = "⚠️ Unsafe to eat."
        elif freshness_score > 0.5:
            age = "3–5 days old 🍂"
            life = "Likely spoiled — bad smell likely."
            health = "❌ May cause food poisoning."
        else:
            age = "5+ days old 💀"
            life = "Fully decayed."
            health = "☠️ Extremely unsafe."
    else:
        if freshness_score > 0.85:
            age = "0–1 day old 🍃"
            life = "Stays fresh for 4–5 more days."
            health = "✅ Excellent & safe."
        elif freshness_score > 0.7:
            age = "1–2 days old 🍏"
            life = "Good for 2–3 more days."
            health = "👍 Still healthy."
        elif freshness_score > 0.55:
            age = "2–3 days old 🌿"
            life = "1–2 days left."
            health = "🟡 Slightly aged but safe."
        else:
            age = "3–4 days old 🧃"
            life = "Near spoilage."
            health = "⚠️ Not recommended."

    return age, life, health, freshness_score, brown_ratio, roughness, vividness


# --------------------------------------------------------
# UI + LOGIC
# --------------------------------------------------------
st.title("🍌 Fruit Freshness Detection System")
st.write("Upload a fruit image to detect **Fresh or Rotten** and get detailed health info.")

uploaded = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])

mode = st.radio("Select Mode", ["🧠 Smart Dynamic Mode", "⚙️ Rule-based Mode"], horizontal=True)

if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Analyze Image"):
        with st.spinner("Analyzing..."):
            try:
                fruit_class, confidence = predict_fruit(img)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.stop()

            st.success("Prediction Complete!")
            st.write(f"### 🏷️ Prediction: **{fruit_class}**")
            st.write(f"### 🎯 Confidence: **{confidence*100:.2f}%**")

            # SMART MODE
            if mode == "🧠 Smart Dynamic Mode":
                age, life, health, score, brown, rough, vivid = analyze_image_advanced(img, fruit_class)

                st.write(f"**Estimated Age:** {age}")
                st.write(f"**Shelf Life:** {life}")
                st.write(f"**Health Info:** {health}")
                st.progress(score)
                st.write(f"🟢 Freshness Score: `{score:.2f}`")
                st.write(f"🟤 Brown Areas: `{brown:.2f}`")
                st.write(f"🌈 Vividness: `{vivid:.2f}`")
                st.write(f"🌾 Roughness: `{rough:.2f}`")

            # RULE-BASED MODE
            else:
                base_conf = confidence
                if "rotten" in fruit_class:
                    if base_conf > 0.85:
                        age = "4–6 days old 🍂"
                        life = "Spoiled — discard."
                        health = "⚠️ Unsafe."
                    elif base_conf > 0.65:
                        age = "3–4 days old 🧃"
                        life = "Likely fermenting."
                        health = "🦠 Not safe."
                    else:
                        age = "2–3 days old ⚠️"
                        life = "Do not consume."
                        health = "❗ Unsafe."
                else:
                    if base_conf > 0.9:
                        age = "0–1 day old 🍃"
                        life = "Stays fresh for days."
                        health = "✅ Very healthy."
                    elif base_conf > 0.75:
                        age = "1–2 days old 🍏"
                        life = "Good for 2–3 days."
                        health = "👍 Healthy."
                    else:
                        age = "2–3 days old 🌿"
                        life = "1–2 days."
                        health = "🟡 Edible but aging."

                st.write(f"**Estimated Age:** {age}")
                st.write(f"**Shelf Life:** {life}")
                st.write(f"**Health Info:** {health}")
