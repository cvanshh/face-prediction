import streamlit as st
import numpy as np
import pickle
from deepface import DeepFace
from PIL import Image
import tempfile

# ---------- PAGE ----------
st.set_page_config(page_title="Celebrity Look-Alike", layout="centered")

st.title("Celebrity Look-Alike Finder")
st.write("Upload a photo to see which celebrity you resemble")

# ---------- LOAD DB ----------
@st.cache_data
def load_db():
    with open("embeddings.pkl", "rb") as f:
        return pickle.load(f)

db = load_db()

# ---------- COSINE ----------
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ---------- PREDICT ----------
def predict(img_path, top_k=3):

    target = DeepFace.represent(
        img_path=img_path,
        model_name='Facenet',
        enforce_detection=False,
        detector_backend='opencv'
    )[0]['embedding']

    target = np.array(target)

    scores = []
    for entry in db:
        emb = np.array(entry["embedding"])
        score = cosine_similarity(target, emb)
        scores.append((entry["name"], score))

    scores.sort(key=lambda x: x[1], reverse=True)

    top_k_results = scores[:top_k]
    best_name, best_score = scores[0]

    if best_score > 0.75:
        confidence = "High"
    elif best_score > 0.55:
        confidence = "Medium"
    else:
        confidence = "Low"

    return best_name, best_score, top_k_results, confidence


# ---------- UPLOAD ----------
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:

    image = Image.open(uploaded_file)
    st.image(image, caption="Your Image", use_column_width=True)

    # Save temp image
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        temp_path = tmp.name

    # ---------- PROCESS ----------
    with st.spinner("Analyzing..."):
        name, score, top_k, confidence = predict(temp_path)

    # ---------- RESULT ----------
    st.subheader(f"Result: {name}")
    st.write(f"Similarity Score: {round(score, 3)}")
    st.write(f"Confidence: {confidence}")
    st.progress(float(score))

    # ---------- TOP MATCHES ----------
    st.subheader("Top Matches")
    for n, s in top_k:
        st.write(f"{n} — {round(s,3)}")
