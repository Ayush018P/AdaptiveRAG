from app.rag.chunker import chunk_text

def test_chunking_length():
    text = "Word " * 1000
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    assert len(chunks[0]['text'].split()) == 100
    
def test_chunking_small_text():
    text = "Short text."
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) == 1
    assert chunks[0]['text'] == "Short text."
