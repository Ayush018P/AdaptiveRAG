from typing import List, Dict

def chunk_text(text: str, chunk_size: int = 100, overlap: int = 20) -> List[Dict]:
    """Splits text into overlapping chunks using word boundaries where possible."""
    words = text.split()
    chunks = []
    
    if not words:
        return chunks
        
    i = 0
    chunk_index = 0
    while i < len(words):
        chunk_words = words[i:i + chunk_size]
        chunk_text = " ".join(chunk_words)
        
        chunks.append({
            "text": chunk_text,
            "chunk_index": chunk_index,
            "page_number": None # Simplification for word-based chunking
        })
        
        i += (chunk_size - overlap)
        chunk_index += 1
        
    return chunks
