import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

# Suppress the HuggingFace Token warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# -----------------------------
# Paths
# -----------------------------
BASE = Path(__file__).resolve().parent.parent
JSON_PATH = BASE / "data" / "extracted" / "metadata.json"
CSV_PATH = BASE / "data" / "input" / "product.csv"
QDRANT_PATH = BASE / "data" / "qdrant_storage"

# -----------------------------
# 1. Initialize Components
# -----------------------------
print("Connecting to Local Qdrant...")
client = QdrantClient(path=str(QDRANT_PATH))
model = SentenceTransformer('BAAI/bge-small-en-v1.5')

COLLECTION_NAME = "compliance_audit"

def build_vector_db():
    try:
        # Check if collection exists instead of using the deprecated recreate_collection
        if client.collection_exists(collection_name=COLLECTION_NAME):
            print(f"Refreshing collection: {COLLECTION_NAME}")
            client.delete_collection(collection_name=COLLECTION_NAME)
        
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )

        points = []
        idx = 0
        print(f"Looking for JSON at: {JSON_PATH} | Found: {JSON_PATH.exists()}")
        print(f"Looking for CSV at: {CSV_PATH} | Found: {CSV_PATH.exists()}")
        # --- A. Process RFQ JSON Chunks ---
        if JSON_PATH.exists():
            with open(JSON_PATH, "r", encoding="utf-8") as f:
                rfq_data = json.load(f)
                items = rfq_data if isinstance(rfq_data, list) else rfq_data.get("text", [])
                
                print(f"Processing {len(items)} RFQ chunks...")
                for item in items:
                    content = item.get("content") or item.get("text")
                    tags = item.get("domain_tags", ["General"])
                    
                    enriched_content = f"[{', '.join(tags)}] {content}"
                    vector = model.encode(enriched_content).tolist()

                    points.append(PointStruct(
                        id=idx,
                        vector=vector,
                        payload={
                            "content": enriched_content,
                            "source": "RFQ",
                            "page": item.get("page_num"),
                            "domain": tags[0] if tags else "General",
                            "type": item.get("type")
                        }
                    ))
                    idx += 1

        # --- B. Process Product CSV Rows ---
        if CSV_PATH.exists():
            df = pd.read_csv(CSV_PATH)
            print(f"Processing {len(df)} Product rows...")
            for _, row in df.iterrows():
                product_desc = (
                    f"[CATALOG] Product ID: {row['product_id']} | Name: {row['product_name']} | "
                    f"Type: {row['product_type']} | Capacity: {row['capacity_cfm']} CFM | "
                    f"ISO: {row['iso_certified']} | Noise: {row['noise_level_db']} dB"
                )
                vector = model.encode(product_desc).tolist()

                points.append(PointStruct(
                    id=idx,
                    vector=vector,
                    payload={
                        "content": product_desc,
                        "source": "CATALOG",
                        "product_id": row['product_id'],
                        "domain": "Product_Spec",
                        "type": "product_spec",
                        "cfm": row.get('capacity_cfm', 0),
                        "iso": row.get('iso_certified', "No")
                    }
                ))
                idx += 1

        # Upsert
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"✅ Qdrant Vector Store ready with {idx} points.")

    finally:
        # Crucial for local databases: Close the client before script ends
        client.close()
        print("📁 Connection closed safely.")

if __name__ == "__main__":
    build_vector_db()