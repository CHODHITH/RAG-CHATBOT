import os
from pypdf import PdfReader

def load_pdf_documents(pdf_paths):
    """
    Loads one or more PDF files and extracts text page by page with metadata.
    Returns a list of dicts: [{'text': ..., 'source': ..., 'page': ...}, ...]
    """
    documents = []
    for path in pdf_paths:
        if not os.path.exists(path):
            continue
        filename = os.path.basename(path)
        try:
            reader = PdfReader(path)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and text.strip():
                    documents.append({
                        "text": text.strip(),
                        "source": filename,
                        "page": page_num + 1
                    })
        except Exception as e:
            print(f"Error reading {filename}: {e}")
    return documents
