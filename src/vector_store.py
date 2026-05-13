import json
import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import PointStruct, VectorParams, Distance

# -----------------------------
# Paths
# -----------------------------
BASE = Path(__file__).resolve().parent.parent
JSON_PATH = BASE / "data" / "extracted" / "metadata.json"
CSV_PATH = BASE / "data" / "input" / "products.csv"
QDRANT_PATH = BASE / "data" / "qdrant_storage"  # Local DB folder

# -----------------------------
# 1. Initialize Qdrant & Model
# -----------------------------
print("Connecting to Local Qdrant...")
client = QdrantClient(path=str(QDRANT_PATH))
model = SentenceTransformer('BAAI/bge-small-en-v1.5')

COLLECTION_NAME = "compliance_audit"

def build_vector_db():
    # Fresh start: Delete and recreate collection
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

    points = []
    idx = 0

    # --- A. Process RFQ JSON Chunks ---
    if JSON_PATH.exists():
        with open(JSON_PATH, "r", encoding="utf-8") as f:
            rfq_data = json.load(f)
            items = rfq_data if isinstance(rfq_data, list) else rfq_data.get("text", [])
            
            print(f"Processing {len(items)} RFQ chunks...")
            for item in items:
                content = item.get("content") or item.get("text")
                tags = item.get("domain_tags", ["General"])
                
                # Metadata Injection (Baking)
                enriched_content = f"[{', '.join(tags)}] {content}"
                vector = model.encode(enriched_content).tolist()

                # Build the 'Point' (Vector + Payload)
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
                    "cfm": row['capacity_cfm'],
                    "iso": row['iso_certified']
                }
            ))
            idx += 1

    # -----------------------------
    # 2. Upsert to Qdrant
    # -----------------------------
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

    print(f"✅ Qdrant Vector Store ready with {idx} points.")
    print(f"📁 Database stored at: {QDRANT_PATH}")

if __name__ == "__main__":
    build_vector_db()