# OFFLINE RAG SYSTEM

A lightweight Retrieval-Augmented Generation (RAG) system implemented in Python.
This project demonstrates a simple offline text retrieval pipeline that provides context-aware prompts for question answering using bag-of-words vectorization and cosine similarity.

## FEATURES

Text Preprocessing: Clean and tokenize raw text.

Chunking: Split documents into overlapping chunks for better retrieval.

Vectorization: Convert text chunks into Bag-of-Words vectors.

Similarity Scoring: Compute cosine similarity between query and document chunks.

Retriever: Retrieve top-k relevant chunks from the document.

RAG Pipeline: Generate an augmented prompt for context-aware answering.

Offline Demo: Interactively ask questions against a local text document.

## HOW IT WORKS

Clean & Tokenize – Converts raw text to lowercase and removes punctuation.

Chunk Text – Splits the document into overlapping chunks (default 20 words per chunk, 5-word overlap).

Vectorize – Uses Bag-of-Words to convert text into numeric vectors.

Compute Similarity – Cosine similarity scores are computed between the query and each chunk.

Retrieve Top Chunks – Returns the most relevant chunks based on similarity.

Generate Augmented Prompt – Combines retrieved chunks with the query for a context-aware prompt.

## References

Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks – Patrick Lewis et al., 2020.
