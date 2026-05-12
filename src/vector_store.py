import json
import pandas as pd
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# -----------------------------
# Paths
# -----------------------------
BASE = Path(__file__).resolve().parent.parent
JSON_PATH = BASE / "data" / "extracted" / "metadata.json"
CSV_PATH = BASE / "data" / "input" / "products.csv"
INDEX_PATH = BASE / "data" / "extracted" / "vector_index.faiss"
METADATA_STORE_PATH = BASE / "data" / "extracted" / "vector_metadata.json"

# -----------------------------
# 1. Load BGE-Small (Top-tier Retrieval Model)
# -----------------------------
print("Loading BGE-Small-EN-v1.5...")
# BGE models work best with a specific instruction prefix for retrieval
model = SentenceTransformer('BAAI/bge-small-en-v1.5')

def build_vector_db():
    all_chunks = []
    all_metadata = []

    # --- A. Process RFQ JSON Chunks ---
    if JSON_PATH.exists():
        with open(JSON_PATH, "r", encoding="utf-8") as f:
            rfq_data = json.load(f)
            items = rfq_data if isinstance(rfq_data, list) else rfq_data.get("text", [])
            
            for item in items:
                content = item.get("content") or item.get("text")
                # We prepend the domain tag to the content to "bake" the physics into the vector
                tags = item.get("domain_tags", ["General"])
                tag_str = f"[{', '.join(tags)}]"
                
                # Metadata Injection: The vector now 'remembers' its domain
                enriched_content = f"{tag_str} {content}"
                
                all_chunks.append(enriched_content)
                all_metadata.append({
                    "source": "RFQ",
                    "page": item.get("page_num"),
                    "domain": tags[0] if tags else "General",
                    "type": item.get("type")
                })

    # --- B. Process Product CSV Rows ---
    if CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH)
        for _, row in df.iterrows():
            # Formatting product data into a highly searchable 'fact sheet'
            product_desc = (
                f"[CATALOG] Product ID: {row['product_id']} | Name: {row['product_name']} | "
                f"Type: {row['product_type']} | Capacity: {row['capacity_cfm']} CFM | "
                f"ISO: {row['iso_certified']} | Noise: {row['noise_level_db']} dB"
            )
            all_chunks.append(product_desc)
            all_metadata.append({
                "source": "CATALOG",
                "product_id": row['product_id'],
                "domain": "Product_Spec",
                "type": "product_spec"
            })

    # -----------------------------
    # 2. Generate Embeddings
    # -----------------------------
    print(f"Embedding {len(all_chunks)} items with instruction-based retrieval...")
    # For BGE, it's a best practice to add a query instruction during search, 
    # but for indexing, we just encode normally.
    embeddings = model.encode(all_chunks, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings).astype('float32')

    # -----------------------------
    # 3. Create FAISS Index
    # -----------------------------
    dimension = embeddings.shape[1]
    # Using IndexFlatIP (Inner Product) because we normalized embeddings
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    # Save
    faiss.write_index(index, str(INDEX_PATH))
    with open(METADATA_STORE_PATH, "w", encoding="utf-8") as f:
        json.dump({"chunks": all_chunks, "metadata": all_metadata}, f, indent=4)

    print(f"✅ Hybrid Vector Store ready: {index.ntotal} entries.")

if __name__ == "__main__":
    build_vector_db()