import streamlit as st
import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from transformers import pipeline

st.set_page_config(page_title="AI Assistant", layout="wide")

st.title("🚀 AI Catalogue Assistant (Debug Mode)")

# =============================pip 
# DEBUG START
# =============================
st.write("✅ App started successfully")

# =============================
# LOAD LIGHT MODEL (SAFE)
# =============================
@st.cache_resource
def load_llm():
    try:
        st.write("🔄 Loading LLM...")
        generator = pipeline("text-generation", model="google/flan-t5-small")
        st.write("✅ LLM Loaded")
        return generator
    except Exception as e:
        st.error(f"❌ LLM Load Error: {e}")
        return None

llm = load_llm()

# =============================
# LOAD EMBEDDING MODEL
# =============================
@st.cache_resource
def load_embedder():
    try:
        st.write("🔄 Loading embedding model...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        st.write("✅ Embedding model loaded")
        return model
    except Exception as e:
        st.error(f"❌ Embedding Error: {e}")
        return None

embed_model = load_embedder()

# =============================
# LOAD FAISS INDEX
# =============================
INDEX_PATH = "data/faiss_index/index.bin"
META_PATH = "data/faiss_index/meta.pkl"

def load_index():
    try:
        if os.path.exists(INDEX_PATH):
            st.write("📦 Loading FAISS index...")
            index = faiss.read_index(INDEX_PATH)

            with open(META_PATH, "rb") as f:
                metadata = pickle.load(f)

            st.write(f"✅ Index loaded with {index.ntotal} items")
            return index, metadata
        else:
            st.warning("⚠️ No FAISS index found. Please build it first.")
            return None, None

    except Exception as e:
        st.error(f"❌ Index Load Error: {e}")
        return None, None

index, metadata = load_index()

# =============================
# CHAT MEMORY
# =============================
if "history" not in st.session_state:
    st.session_state.history = ""

# =============================
# RETRIEVAL FUNCTION
# =============================
def retrieve(query):
    try:
        query_emb = embed_model.encode([query], normalize_embeddings=True)
        scores, ids = index.search(query_emb, 3)
        return [metadata[i] for i in ids[0]]
    except Exception as e:
        st.error(f"❌ Retrieval Error: {e}")
        return []

# =============================
# GENERATE ANSWER
# =============================
def generate_answer(query, context):
    try: 
        prompt = f"""
        Answer the question based on context.

        Context:
        {context}

        Question:
        {query}
        """

        result = llm(prompt, max_length=150)[0]["generated_text"]
        return result

    except Exception as e:
        return f"❌ LLM Error: {e}"

# =============================
# CHAT UI
# =============================
query = st.chat_input("Ask something...")

if query:
    st.write(f"🧑 User: {query}")

    if index is None:
        st.error("❌ No index loaded. Cannot search.")
    else:
        results = retrieve(query)

        if results:
            context = "\n\n".join([
                f"{r['title']} {r['description']}"
                for r in results
            ])

            st.write("📚 Retrieved Context:")
            st.write(context)

            answer = generate_answer(query, context)

            st.write("🤖 Answer:")
            st.write(answer)

            # Save memory
            st.session_state.history += f"\nUser: {query}\nAI: {answer}\n"

        else:
            st.warning("⚠️ No results found")


print("version 2")