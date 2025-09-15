import os
import numpy as np
import faiss
import pickle
from PIL import Image
import streamlit as st
from sentence_transformers import SentenceTransformer

# Folders + paths
IMAGE_FOLDER = "images"
INDEX_PATH = "faiss.index"
META_PATH = "meta.pkl"
MODEL_NAME = "clip-ViT-B-32"

os.makedirs(IMAGE_FOLDER, exist_ok=True)

# Load model
@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_NAME)

model = load_model()

# Load index + meta if exist
if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
else:
    index = None
    meta = []

# Helper functions
def save_index():
    if index is None:
        return
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(meta, f)

def image_to_embedding(image: Image.Image):
    img = image.convert("RGB")
    emb = model.encode([img], convert_to_numpy=True)[0]
    emb = emb / (np.linalg.norm(emb) + 1e-10)  # normalize
    return emb.astype("float32")

def ensure_index(dim):
    global index
    if index is None:
        index = faiss.IndexFlatIP(dim)  # cosine similarity

def add_image(filepath):
    global meta
    image = Image.open(filepath)
    emb = image_to_embedding(image)
    ensure_index(emb.shape[0])
    index.add(emb.reshape(1, -1))
    new_id = len(meta)
    meta.append({"filename": os.path.basename(filepath), "id": new_id})
    save_index()
    return new_id

def search_image(image, top_k=5):
    emb = image_to_embedding(image).reshape(1, -1)
    if index is None or index.ntotal == 0:
        return []
    D, I = index.search(emb, top_k)
    results = []
    for dist, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        meta_item = meta[idx]
        results.append({"filename": meta_item["filename"], "score": float(dist)})
    return results

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🔎 Image Similarity Finder")

# Upload for indexing
st.header("📥 Add Images to Index")
uploaded_files = st.file_uploader("Upload images", type=["jpg","png","jpeg"], accept_multiple_files=True)
if uploaded_files:
    for uf in uploaded_files:
        path = os.path.join(IMAGE_FOLDER, uf.name)
        with open(path, "wb") as f:
            f.write(uf.read())
        add_image(path)
    st.success(f"Indexed {len(uploaded_files)} images ✅")

# Upload for search
st.header("🔍 Search by Image")
query_file = st.file_uploader("Upload a query image", type=["jpg","png","jpeg"], key="query")
if query_file:
    query_image = Image.open(query_file)
    st.image(query_image, caption="Query Image", width=250)
    results = search_image(query_image, top_k=5)
    st.subheader("Results")
    for r in results:
        img_path = os.path.join(IMAGE_FOLDER, r["filename"])
        if os.path.exists(img_path):
            st.image(img_path, caption=f"{r['filename']} (score={r['score']:.4f})", width=200)
