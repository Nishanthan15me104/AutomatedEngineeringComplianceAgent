import fitz  # PyMuPDF
import pdfplumber
import json
import torch
import io
import re
from PIL import Image
from pathlib import Path
from transformers import BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer, util

# -----------------------------
# Paths & Config
# -----------------------------
BASE = Path(__file__).resolve().parent.parent
PDF_PATH = BASE / "data" / "input" / "sample_rfq.pdf"
OUT_IMG_DIR = BASE / "data" / "extracted" / "images"
OUT_JSON = BASE / "data" / "extracted" / "metadata.json"

OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

# NEW: Chunking config
CHUNK_SIZE = 500  # Words
CHUNK_OVERLAP = 50 

# -----------------------------
# 1. Logic Helpers (Cleaning & Chunking)
# -----------------------------

def clean_text(text):
    """Removes redundant whitespace and common RFQ boilerplate headers/footers."""
    if not text: return ""
    # Remove multiple newlines/spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove generic footer patterns (e.g., "Page 1 of 20")
    text = re.sub(r'(?i)page \d+ of \d+', '', text)
    return text

def chunk_text(text, max_words=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Splits long text into overlapping chunks to maintain context."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words - overlap):
        chunk = " ".join(words[i:i + max_words])
        if len(chunk) > 50: # Ignore tiny fragments
            chunks.append(chunk)
    return chunks

def is_table_continuation(current_table, last_entry):
    """Heuristic: Checks if current table is a continuation of the last page's table."""
    if not last_entry or last_entry['type'] != 'table':
        return False
    # If column counts match, it's likely a continuation
    if len(current_table[0]) == last_entry.get('col_count'):
        return True
    return False

# -----------------------------
# 2. ML Models (CLIP & BLIP)
# -----------------------------
print("Loading Models...")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
clip_model = SentenceTransformer('clip-ViT-B-32').to(device)

CATEGORIES = [
    "Structural dimensions, physical size, steel, and heavy weight", 
    "Thermal cooling, heat dissipation, and high temperature", 
    "Fluid flow, liquid ventilation, and air pressure",
    "General business terms, commercial clauses, and conditions"
]
CATEGORY_EMBEDDINGS = clip_model.encode(CATEGORIES, convert_to_tensor=True)
TAG_MAP = {0: "Structural", 1: "Thermal", 2: "Flow", 3: "General"}

def get_semantic_tag(content):
    with torch.no_grad():
        content_emb = clip_model.encode(content, convert_to_tensor=True)
        cosine_scores = util.cos_sim(content_emb, CATEGORY_EMBEDDINGS)[0]
        best_idx = torch.argmax(cosine_scores).item()
        return [TAG_MAP[best_idx]]

def generate_image_caption(pil_image):
    with torch.no_grad():
        inputs = blip_processor(pil_image, return_tensors="pt").to(device)
        output = blip_model.generate(**inputs)
        return blip_processor.decode(output[0], skip_special_tokens=True)

# -----------------------------
# 3. Enhanced Extraction Pipeline
# -----------------------------
def process_pdf(pdf_path):
    extracted_data = []
    doc = fitz.open(pdf_path)
    plumber_doc = pdfplumber.open(pdf_path)

    for page_num in range(len(doc)):
        page = doc[page_num]
        actual_page = page_num + 1
        print(f"Processing Page {actual_page}...")

        # --- A. Text Chunking & Cleaning ---
        raw_text = page.get_text("text")
        cleaned_text = clean_text(raw_text)
        text_chunks = chunk_text(cleaned_text)
        
        for chunk in text_chunks:
            extracted_data.append({
                "type": "text",
                "content": chunk,
                "page_num": actual_page,
                "domain_tags": get_semantic_tag(chunk)
            })

        # --- B. Table Handling (with Continuation Logic) ---
        plumber_page = plumber_doc.pages[page_num]
        tables = plumber_page.extract_tables()
        
        for table in tables:
            # Clean empty cells and convert to string
            table_str = "\n".join([" | ".join([str(cell) if cell else "" for cell in row]) for row in table])
            table_str = clean_text(table_str)
            
            if not table_str: continue

            # Check if we should merge with last entry
            if extracted_data and is_table_continuation(table, extracted_data[-1]):
                extracted_data[-1]['content'] += "\n" + table_str
                # Re-tag the merged content
                extracted_data[-1]['domain_tags'] = get_semantic_tag(extracted_data[-1]['content'])
            else:
                extracted_data.append({
                    "type": "table",
                    "content": table_str,
                    "col_count": len(table[0]),
                    "page_num": actual_page,
                    "domain_tags": get_semantic_tag(table_str)
                })

        # --- C. Image Captioning ---
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            pil_image = Image.open(io.BytesIO(base_image["image"])).convert("RGB")
            
            caption = generate_image_caption(pil_image)
            image_tags = get_semantic_tag(pil_image)
            
            img_filename = f"img_{actual_page}_{img_index}.png"
            pil_image.save(OUT_IMG_DIR / img_filename)

            extracted_data.append({
                "type": "image",
                "content": f"Image showing: {caption}",
                "image_file": img_filename,
                "page_num": actual_page,
                "domain_tags": image_tags
            })

    # Save to JSON
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(extracted_data, f, indent=4)
        
    print(f"✅ Success. {len(extracted_data)} semantic units extracted.")

if __name__ == "__main__":
    process_pdf(PDF_PATH)