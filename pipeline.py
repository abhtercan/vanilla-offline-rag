from chunker import chunk_text
from vectorizer import build_vocab
from retriever import retrieve

class RAGPipeline:

    def __init__(self, document):
        self.chunks = chunk_text(document)
        self.vocab = build_vocab(self.chunks)

    def generate(self, query, top_k=2):
        top_chunks = retrieve(query, self.chunks, self.vocab, top_k)

        context = top_chunks[0][0] if top_chunks else None

        augmented_prompt = f"""

Question:
{query}

Context:
{context}

Answer based only on the context above.
"""
        return augmented_prompt, top_chunks
