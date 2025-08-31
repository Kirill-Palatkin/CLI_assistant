import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import build_chunks_from_docs, save_pickle

from dotenv import dotenv_values

config = dotenv_values()


def build_index(docs_dir: str, index_path: str, chunk_size: int = 150, overlap: int = 50):
    """
    Строит TF-IDF индекс по документам и сохраняет в index.pkl
    """
    print(f"[index] Чтение документов из {docs_dir} ...")
    chunks = build_chunks_from_docs(docs_dir, chunk_size=chunk_size, overlap=overlap)
    texts = [c["text"] for c in chunks]

    if not texts:
        raise ValueError("Не найдено ни одного чанка для индексирования")

    print(f"[index] Построение TF-IDF для {len(texts)} чанков ...")
    vectorizer = TfidfVectorizer(stop_words=None)  # можно добавить stop_words="english" если нужно
    tfidf_matrix = vectorizer.fit_transform(texts)

    index_data = {
        "vectorizer": vectorizer,
        "tfidf_matrix": tfidf_matrix,
        "chunks": chunks,
    }

    save_pickle(index_data, index_path)
    print(f"[index] Индекс сохранён в {index_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Построение TF-IDF индекса")
    parser.add_argument("--docs_dir", default=config.get("DOCS_DIR", "./docs"))
    parser.add_argument("--index_path", default=config.get("INDEX_PATH", "./index.pkl"))
    parser.add_argument("--chunk_size", type=int, default=int(config.get("CHUNK_SIZE", "150")))
    parser.add_argument("--overlap", type=int, default=int(config.get("CHUNK_OVERLAP", "50")))
    args = parser.parse_args()

    build_index(
        docs_dir=args.docs_dir,
        index_path=args.index_path,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
