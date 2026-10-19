import os
from typing import List
from PyPDF2 import PdfReader
from docx import Document

def load_document(file_path: str, doc_type: str) -> str:
    """Loads text from different document types."""
    if doc_type == 'application/pdf' or file_path.endswith('.pdf'):
        return load_pdf(file_path)
    elif doc_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' or file_path.endswith('.docx'):
        return load_docx(file_path)
    elif doc_type == 'text/plain' or file_path.endswith('.txt') or file_path.endswith('.md'):
        return load_txt(file_path)
    else:
        raise ValueError(f"Unsupported document type: {doc_type}")

def load_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n\n"
    return text

def load_docx(file_path: str) -> str:
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def load_txt(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()
