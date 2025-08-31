import os
from typing import List, Dict, Optional
from dotenv import load_dotenv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from utils import load_pickle, citation_tag


class Retriever:
    def __init__(self, index_path: str, top_k: int = 3, threshold: float = 0.42):
        index_data = load_pickle(index_path)
        self.vectorizer = index_data["vectorizer"]
        self.tfidf_matrix = index_data["tfidf_matrix"]
        self.chunks = index_data["chunks"]
        self.top_k = top_k
        self.threshold = threshold

    def retrieve(self, query: str) -> Optional[List[Dict]]:
        """
        Возвращает список top-k чанков (dict) с метаинформацией и score.
        Если лучший скор < threshold, вернуть None.
        """
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.tfidf_matrix)[0]

        # argsort по убыванию
        top_idx = np.argsort(sims)[::-1][:self.top_k]
        top_scores = sims[top_idx]

        if top_scores[0] < self.threshold:
            return None

        results = []
        for idx, score in zip(top_idx, top_scores):
            meta = self.chunks[idx].copy()
            meta["score"] = float(score)
            meta["citation"] = citation_tag(meta)
            results.append(meta)

        return results


if __name__ == "__main__":
    # Тестовый запуск
    load_dotenv()
    index_path = os.getenv("INDEX_PATH", "./index.pkl")
    top_k = int(os.getenv("TOP_K", "3"))
    threshold = float(os.getenv("NO_ANSWER_THRESHOLD", "0.42"))

    retriever = Retriever(index_path, top_k=top_k, threshold=threshold)

    while True:
        try:
            query = input("Введите запрос (или 'exit'): ").strip()
            if query.lower() in {"exit", "quit"}:
                break
            results = retriever.retrieve(query)
            if results is None:
                print("⚠️  Нет ответа (низкая релевантность)")
            else:
                for r in results:
                    print(f"[{r['score']:.3f}] {r['citation']}")
                    print(f"    {r['text'][:500]}...\n")
        except KeyboardInterrupt:
            break
