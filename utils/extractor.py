import os
import re
from typing import List, Dict
from pypdf import PdfReader


def extract_products(pdf_path: str, filename: str) -> List[Dict[str, object]]:
    """Extract a simple list of product-like entries from a PDF catalog."""
    try:
        reader = PdfReader(pdf_path)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        print(f"⚠️ Could not read {filename}: {e}")
        return []

    if not text.strip():
        return []

    # Split by lines and keep entries that look like products.
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    products: List[Dict[str, object]] = []

    current_title = None
    current_desc_parts = []
    current_features = []

    for line in lines:
        if len(line) < 4:
            continue

        if re.search(r"\b(Series|Model|Product|Item|Catalog|System|Unit|Panel|Battery|Inverter)\b", line, re.I):
            if current_title:
                products.append({
                    "title": current_title,
                    "description": " ".join(current_desc_parts).strip() or "No description available",
                    "features": current_features,
                })
            current_title = line
            current_desc_parts = []
            current_features = []
        elif current_title:
            if re.search(r"^(?:Features|Key Features|Advantages|Benefits|Specifications):", line, re.I):
                current_features.append(line)
            elif len(current_desc_parts) < 6:
                current_desc_parts.append(line)

    if current_title:
        products.append({
            "title": current_title,
            "description": " ".join(current_desc_parts).strip() or "No description available",
            "features": current_features,
        })

    if not products:
        # Fallback: create one product entry from the whole extracted text.
        products.append({
            "title": os.path.splitext(filename)[0],
            "description": text[:1200],
            "features": ["Extracted from PDF catalog"],
        })

    return products
