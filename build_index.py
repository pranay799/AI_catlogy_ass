import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from utils.extractor import extract_products  # use your extractor

PDF_DIR = "data/pdfs"
INDEX_PATH = "data/faiss_index/index.bin"
META_PATH = "data/faiss_index/meta.pkl"

os.makedirs("data/faiss_index", exist_ok=True)

print("📂 Reading PDFs...")

all_products = []

for file in os.listdir(PDF_DIR):
    path = os.path.join(PDF_DIR, file)
    print(f"Processing: {file}")
    products = extract_products(path, file)
    all_products.extend(products)

print(f"✅ Total products: {len(all_products)}")

# ------------------------
# Embeddings
# ------------------------
print("🔄 Creating embeddings...")

model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [
    f"{p['title']} {p['description']} {' '.join(p['features'])}"
    for p in all_products
]

embeddings = model.encode(texts, normalize_embeddings=True)

# ------------------------
# FAISS Index
# ------------------------
print("⚡ Building FAISS index...")

dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings)

# ------------------------
# Save
# ------------------------
print("💾 Saving index...")

faiss.write_index(index, INDEX_PATH)

with open(META_PATH, "wb") as f:
    pickle.dump(all_products, f)

print("🎉 Index built successfully!")