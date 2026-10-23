from main import chunk_text 
def test_chunk_text_returns_multiple_chunks():
    text = "A"*1000 
    chunks = chunk_text(text, chunk_size=500, overlap=100)
    assert len(chunks) == 3
def test_chunk_text_preserves_overlap(): 
    text = "A"*1000 
    chunks = chunk_text(text, chunk_size=500, overlap=100)
    assert chunks[0][-100:] == chunks[1][:100]
def test_chunk_text_returns_empty_list_for_empty_text():
    chunks = chunk_text("")

    assert chunks == []