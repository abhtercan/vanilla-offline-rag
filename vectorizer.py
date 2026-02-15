from collections import Counter
from chunker import tokenize

def build_vocab(chunks):
    vocab = set()
    for chunk in chunks:
        vocab.update(tokenize(chunk))
    return list(vocab)

def text_to_vector(text, vocab):
    words = tokenize(text)
    word_count = Counter(words)
    return [word_count.get(word, 0) for word in vocab]
