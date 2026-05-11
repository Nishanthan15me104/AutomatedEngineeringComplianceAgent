import fitz  # PyMuPDF
import pdfplumber
import json
import torch
import io
from PIL import Image
from pathlib import Path
from transformers import BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer, util

# -----------------------------
# Paths
# -----------------------------
BASE = Path(__file__).resolve().parent.parent
PDF_PATH = BASE / "data" / "input" / "sample_rfq.pdf"
OUT_IMG_DIR = BASE / "data" / "extracted" / "images"
OUT_JSON = BASE / "data" / "extracted" / "metadata.json"

OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# 1. Load ML Models
# -----------------------------
print("Loading BLIP for Image Captioning...")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
blip_model.eval()

print("Loading CLIP for Semantic Tagging...")
# CLIP can encode both text and images into the same vector space
clip_model = SentenceTransformer('clip-ViT-B-32').to(device)

# -----------------------------
# 2. Semantic Tagging Logic (Replaces Keyword Heuristics)
# -----------------------------
# We define our categories with descriptive phrases so CLIP understands the context
CATEGORIES = [
    "Structural dimensions, physical size, steel, and heavy weight", 
    "Thermal cooling, heat dissipation, and high temperature", 
    "Fluid flow, liquid ventilation, and air pressure",
    "General business terms and conditions"
]
# Pre-compute the embeddings for our categories
CATEGORY_EMBEDDINGS = clip_model.encode(CATEGORIES, convert_to_tensor=True)

# Map the long descriptions back to clean tags
TAG_MAP = {
    0: "Structural",
    1: "Thermal",
    2: "Flow",
    3: "General"
}

def get_semantic_tag(content):
    """
    Passes text, tables, or PIL Images to CLIP and finds the closest matching domain.
    """
    with torch.no_grad():
        content_emb = clip_model.encode(content, convert_to_tensor=True)
        # Calculate cosine similarity between the content and our 4 categories
        cosine_scores = util.cos_sim(content_emb, CATEGORY_EMBEDDINGS)[0]
        best_idx = torch.argmax(cosine_scores).item()
        
        # Optional: Add a threshold. If the highest score is very low, default to General.
        if cosine_scores[best_idx] < 0.20:
            return ["General"]
            
        return [TAG_MAP[best_idx]]

def generate_image_caption(pil_image):
    """Uses BLIP to generate a text description of the image."""
    with torch.no_grad():
        inputs = blip_processor(pil_image, return_tensors="pt").to(device)
        output = blip_model.generate(**inputs)
        return blip_processor.decode(output[0], skip_special_tokens=True)

# -----------------------------
# 3. Extraction Pipeline
# -----------------------------
def process_pdf(pdf_path):
    extracted_data = []
    
    doc = fitz.open(pdf_path)
    plumber_doc = pdfplumber.open(pdf_path)

    for page_num in range(len(doc)):
        page = doc[page_num]
        actual_page = page_num + 1
        print(f"Processing Page {actual_page}...")

        # --- A. Extract & Tag Text ---
        text = page.get_text("text").strip()
        if text:
            extracted_data.append({
                "type": "text",
                "text": text,
                "page_num": actual_page,
                "domain_tags": get_semantic_tag(text)
            })

        # --- B. Extract & Tag Tables ---
        plumber_page = plumber_doc.pages[page_num]
        tables = plumber_page.extract_tables()
        for i, table in enumerate(tables):
            table_str = "\n".join([" | ".join([str(cell) if cell else "" for cell in row]) for row in table])
            extracted_data.append({
                "type": "table",
                "text": table_str,
                "page_num": actual_page,
                "domain_tags": get_semantic_tag(table_str)
            })

        # --- C. Extract, Caption, & Tag Images ---
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            
            # Convert bytes to PIL Image for the ML models
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            
            # 1. Get Caption via BLIP
            caption = generate_image_caption(pil_image)
            
            # 2. Get Tag via CLIP (Feeding the actual image, not the text!)
            image_tags = get_semantic_tag(pil_image)
            
            # Save image to directory
            img_filename = f"img_{actual_page}_{img_index}.{image_ext}"
            img_filepath = OUT_IMG_DIR / img_filename
            pil_image.save(img_filepath)

            # 3. Add the image metadata to our JSON so the LLM can "read" it later
            extracted_data.append({
                "type": "image",
                "text": f"[Image Caption]: {caption}", # The caption acts as the text payload
                "image_file": img_filename,
                "page_num": actual_page,
                "domain_tags": image_tags
            })

    # Save Metadata JSON
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"text": extracted_data}, f, indent=4)
        
    print(f"✅ Extraction Complete! Saved to {OUT_JSON}")
    print(f"✅ Extracted Images saved to {OUT_IMG_DIR}")

if __name__ == "__main__":
    if PDF_PATH.exists():
        process_pdf(PDF_PATH)
    else:
        print(f"❌ PDF not found at {PDF_PATH}. Please add a sample PDF.")