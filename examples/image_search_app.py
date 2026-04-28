import os
import sys
import asyncio
import streamlit as st
from PIL import Image

# Add the parent directory to sys.path to allow local import of sumospace
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sumospace.ingest import UniversalIngestor

# Setup the page
st.set_page_config(page_title="Sumo Image Space", page_icon="🖼️", layout="wide")

IMAGE_DIR = "uploaded_images"
os.makedirs(IMAGE_DIR, exist_ok=True)

@st.cache_resource
def get_ingestor():
    # Initialize the UniversalIngestor using the CLIP model for vision
    ingestor = UniversalIngestor(
        chroma_path=".sumo_image_db",
        collection_name="image_space",
        embedding_provider="local",
        embedding_model="clip-ViT-B-32"
    )
    # Run the async initialization
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    loop.run_until_complete(ingestor.initialize())
    return ingestor

ingestor = get_ingestor()

st.title("🖼️ Sumo Image Embedding Space")
st.markdown("Build a visual embedding space with `sumospace` by uploading images, then search for similar ones using a query image.")

# --- Sidebar for Data Ingestion ---
st.sidebar.header("1. Ingest Images")
st.sidebar.write("Upload a bunch of images to populate the ChromaDB space.")
uploaded_files = st.sidebar.file_uploader("Upload library images", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True)

if st.sidebar.button("Ingest Images"):
    if uploaded_files:
        with st.spinner(f"Ingesting {len(uploaded_files)} images into the space..."):
            images = []
            ids = []
            metadatas = []
            for f in uploaded_files:
                # Save locally so we can display them later
                file_path = os.path.join(IMAGE_DIR, f.name)
                with open(file_path, "wb") as out_f:
                    out_f.write(f.getbuffer())
                
                # Load image for embedding
                img = Image.open(file_path).convert("RGB")
                images.append(img)
                ids.append(f.name)
                metadatas.append({"path": file_path, "filename": f.name})
            
            # Embed using the local CLIP model
            # sumospace's LocalEmbeddingProvider wraps sentence-transformers, which accepts PIL Images directly
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            embeddings = loop.run_until_complete(ingestor._embedder.embed(images))
            
            # Upsert into ChromaDB via sumospace
            ingestor._collection.upsert(
                ids=ids,
                documents=ids, # Using filenames as documents
                embeddings=embeddings,
                metadatas=metadatas
            )
        st.sidebar.success(f"Successfully ingested {len(uploaded_files)} images!")
    else:
        st.sidebar.warning("Please upload some images first.")

# --- Main Area for Querying ---
st.header("2. Search Similar Images")
query_file = st.file_uploader("Upload Query Image", type=["png", "jpg", "jpeg", "webp"])

if query_file:
    query_img = Image.open(query_file).convert("RGB")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(query_img, caption="Query Image", use_container_width=True)
    
    if st.button("Find Top 5 Similar Images", type="primary"):
        with st.spinner("Searching space..."):
            # Ensure the collection is not empty
            if ingestor._collection.count() == 0:
                st.error("The embedding space is empty. Please ingest images first via the sidebar.")
            else:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                q_emb = loop.run_until_complete(ingestor._embedder.embed([query_img]))
                
                n_results = min(5, ingestor._collection.count())
                results = ingestor._collection.query(
                    query_embeddings=q_emb,
                    n_results=n_results,
                    include=["metadatas", "distances"]
                )
                
                if results and results["metadatas"][0]:
                    st.subheader(f"Top {n_results} Results")
                    res_cols = st.columns(n_results)
                    for i in range(n_results):
                        meta = results["metadatas"][0][i]
                        dist = results["distances"][0][i]
                        path = meta["path"]
                        
                        with res_cols[i]:
                            if os.path.exists(path):
                                res_img = Image.open(path)
                                st.image(res_img, caption=f"{meta['filename']}\nDist: {dist:.3f}", use_container_width=True)
                            else:
                                st.error(f"Image not found: {meta['filename']}")
