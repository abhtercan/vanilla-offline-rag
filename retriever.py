from vectorizer import text_to_vector
from similarity import cosine_similarity

def retrieve(query, chunks, vocab, top_k=2):
    query_vec = text_to_vector(query, vocab)

    scores = []

    for chunk in chunks:
        chunk_vec = text_to_vector(chunk, vocab)
        score = cosine_similarity(query_vec, chunk_vec)
        scores.append((chunk, score))

    scores.sort(key=lambda x: x[1], reverse=True)

    return scores[:top_k]
