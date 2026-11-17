import json
import sys

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import CHROMA_DIR, EMBEDDING_MODEL, EVAL_DIR, CHUNKS_FILE, GOLD_FILE, RESULTS_FILE

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    return Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings
    )


def export_chunks(vectorstore):
    data = vectorstore.get(
        include=["documents", "metadatas"]
    )

    chunks = []

    for chunk_id, text, metadata in zip(
        data["ids"],
        data["documents"],
        data["metadatas"]
    ):
        chunks.append({
            "id": chunk_id,
            "text": text,
            "metadata": metadata
        })

    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    with open(CHUNKS_FILE, "w", encoding="utf-8") as file:
        json.dump(chunks, file, ensure_ascii=False, indent=2)

    print(f"Wyeksportowano {len(chunks)} fragmentów.")


def recall_at_k(retrieved_ids, relevant_ids):
    relevant = set(relevant_ids)

    if not relevant:
        raise ValueError("Brak oznaczonych fragmentów")

    found = set(retrieved_ids) & relevant

    return len(found) / len(relevant)


def evaluate_retrieval(vectorstore):
    with open(GOLD_FILE, encoding="utf-8") as file:
        cases = json.load(file)["cases"]

    if not cases:
        raise ValueError("Brak przypadków testowych")

    available_ids = set(vectorstore.get(include=[])["ids"])
    results = []

    for case in cases:
        question = case["question"]
        relevant_ids = case["relevant_chunk_ids"]

        if not relevant_ids or not set(relevant_ids).issubset(available_ids):
            raise ValueError(
                f"Not correct relevant_chunk_ids: {case['id']}"
            )

        docs = vectorstore.similarity_search(question, k=5)
        retrieved_ids = [doc.id for doc in docs]

        if any(chunk_id is None for chunk_id in retrieved_ids):
            raise ValueError("The search results don't have identificators")

        result = {
            "id": case["id"],
            "question": question,
            "retrieved_ids": retrieved_ids,
            "recall@2": recall_at_k(retrieved_ids[:2], relevant_ids),
            "recall@3": recall_at_k(retrieved_ids[:3], relevant_ids),
            "recall@5": recall_at_k(retrieved_ids[:5], relevant_ids)
        }

        results.append(result)

        print(
            f"{case['id']}: "
            f"R@2={result['recall@2']:.2f} | "
            f"R@3={result['recall@3']:.2f} | "
            f"R@5={result['recall@5']:.2f}"
        )

    summary = {
        metric: round(
            sum(r[metric] for r in results) / len(results) * 100,
            2
        )
        for metric in ("recall@2", "recall@3", "recall@5")
    }

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_FILE, "w", encoding="utf-8") as file:
        json.dump(
            {"summary_percent": summary, "results": results},
            file,
            ensure_ascii=False,
            indent=2
        )

    print("\nŚredni Recall (%):", summary)


if __name__ == "__main__":
    vectorstore = load_vectorstore()

    if len(sys.argv) != 2:
        raise SystemExit(
            "Use: python -m evaluation.retrieval export|evaluate"
        )

    if sys.argv[1] == "export":
        export_chunks(vectorstore)

    elif sys.argv[1] == "evaluate":
        evaluate_retrieval(vectorstore)

    else:
        raise SystemExit("Unknown command")