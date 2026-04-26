import streamlit as st
from PIL import Image
import tempfile, pickle, os
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from collections import Counter, defaultdict

st.set_page_config(page_title="Celebrity Face Recognition", layout="centered")

# ---- Load DB ----
@st.cache_resource
def load_db():
    with open("embeddings.pkl", "rb") as f:
        return pickle.load(f)
db = load_db()

# ---- Centroids ----
@st.cache_resource
def build_centroids(db):
    groups = defaultdict(list)
    for e in db:
        groups[e["name"]].append(np.array(e["embedding"]))
    return {k: np.mean(v, axis=0) for k, v in groups.items()}
centroids = build_centroids(db)

# ---- Models (lightweight) ----
device = 'cpu'
@st.cache_resource
def load_models():
    mtcnn = MTCNN(image_size=160, margin=20, device=device)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    return mtcnn, resnet

mtcnn, resnet = load_models()

def get_embedding(img: Image.Image):
    face = mtcnn(img)
    if face is None:
        # fallback: use whole image resized
        img = img.resize((160,160))
        face = torch.tensor(np.array(img).transpose(2,0,1)).float()/255.0
    if face.ndim == 3:
        face = face.unsqueeze(0)
    with torch.no_grad():
        emb = resnet(face.to(device)).cpu().numpy()[0]
    return emb

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def predict_from_image(image: Image.Image, threshold=0.55, top_k=5):
    target = get_embedding(image)

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

# ---- UI ----
st.title("Celebrity Face Recognition")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((160,160))

    name, score, top = predict_from_image(image)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Input")
        st.image(image)
    with col2:
        st.subheader("Prediction")
        st.write(name)

    st.write(f"**Confidence:** {score:.2f}")
    st.write("Top Matches:")
    for n, s in top[:3]:
        st.write(f"{n} — {s:.2f}")
        
