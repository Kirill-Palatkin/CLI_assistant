import argparse
from retriever import Retriever
from llm_stub import generate_answer

from dotenv import dotenv_values

config = dotenv_values()


def main():

    index_path = config.get("INDEX_PATH", "./index.pkl")
    top_k = int(config.get("TOP_K", "3"))
    threshold = float(config.get("NO_ANSWER_THRESHOLD", "0.42"))

    retriever = Retriever(index_path, top_k=top_k, threshold=threshold)

    parser = argparse.ArgumentParser(description="RAG CLI ассистент")
    parser.add_argument("--query", type=str, required=False, help="Вопрос для ассистента")
    args = parser.parse_args()

    if args.query:
        query = args.query
    else:
        query = input("Введите запрос: ").strip()

    results = retriever.retrieve(query)
    if results is None:
        print("⚠️ Нет ответа (релевантность ниже порога).")
        return

    answer = generate_answer(query, results)
    print("\n=== Ответ ===")
    print(answer)
    print("\n=== Цитаты ===")
    for r in results:
        print(f"[{r['score']:.3f}] {r['citation']}")
        print(f"    {r['text'][:500]}...\n")


if __name__ == "__main__":
    main()
