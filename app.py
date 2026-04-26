import streamlit as st
from PIL import Image
import tempfile
import pickle
import numpy as np
from deepface import DeepFace
from collections import Counter, defaultdict

# Load embeddings
with open("embeddings.pkl", "rb") as f:
    db = pickle.load(f)

# Build centroids
groups = defaultdict(list)
for e in db:
    groups[e["name"]].append(np.array(e["embedding"]))
centroids = {k: np.mean(v, axis=0) for k, v in groups.items()}

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def predict(img_path, threshold=0.55, top_k=5):
    rep = DeepFace.represent(
        img_path=img_path,
        model_name='Facenet',
        enforce_detection=False,
        detector_backend='opencv'
    )
    target = np.array(rep[0]['embedding'])

    scores = []
    for entry in db:
        s = cosine_similarity(target, entry["embedding"])
        scores.append((entry["name"], float(s)))

    scores.sort(key=lambda x: x[1], reverse=True)

    top_k_results = scores[:top_k]
    names = [n for n, _ in top_k_results]
    vote = Counter(names).most_common(1)[0][0]

    centroid_score = cosine_similarity(target, centroids[vote])

    if centroid_score < threshold:
        return "Unknown", centroid_score, top_k_results

    return vote, centroid_score, top_k_results

# UI
st.title("Celebrity Face Recognition")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((160,160))

    st.image(image)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name, format="JPEG")

        name, score, top = predict(tmp.name)

    st.write(f"Prediction: {name}")
    st.write(f"Confidence: {score:.2f}")

    st.write("Top Matches:")
    for n, s in top[:3]:
        st.write(f"{n} - {s:.2f}")
