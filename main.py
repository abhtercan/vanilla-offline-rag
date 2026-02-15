from pipeline import RAGPipeline

def load_document(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":

    document = load_document("data.txt")
    rag = RAGPipeline(document)

    print("Offline Modular RAG System")
    print("----------------------------")

    while True:
        query = input("\nAsk a question (type 'exit' to quit): ")

        if query.lower() == "exit":
            break

        augmented_prompt, retrieved = rag.generate(query)

        print("\nTop Retrieved Chunks:")
        for chunk, score in retrieved:
            print(f"Score: {round(score, 4)} | {chunk}")

        print("\n--- AUGMENTED PROMPT ---")
        print(augmented_prompt)
