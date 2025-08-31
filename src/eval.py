import os
import json
from dotenv import load_dotenv

from retriever import Retriever
from llm_stub import generate_answer


def load_questions(path: str):
    questions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    return questions


def evaluate(questions_path: str):
    load_dotenv()
    index_path = os.getenv("INDEX_PATH", "./index.pkl")
    top_k = int(os.getenv("TOP_K", "3"))
    threshold = float(os.getenv("NO_ANSWER_THRESHOLD", "0.42"))

    retriever = Retriever(index_path, top_k=top_k, threshold=threshold)
    questions = load_questions(questions_path)

    total = len(questions)
    correct = 0

    for q in questions:
        query = q["question"]
        gold_phrases = q["gold_phrases"]

        results = retriever.retrieve(query)
        if results is None:
            answer = ""
        else:
            answer = generate_answer(query, results)

        # Проверка: есть ли хотя бы одна из gold_phrases в ответе
        if any(phrase.lower() in answer.lower() for phrase in gold_phrases):
            correct += 1
            status = "✅"
        else:
            status = "❌"

        print(f"{status} Q: {query}")
        print(f"   Answer: {answer[:220]}...")
        print(f"   Gold: {gold_phrases}\n")

    accuracy = correct / total if total else 0.0
    print(f"=== Итоговая точность: {accuracy*100:.1f}% ({correct}/{total}) ===")


if __name__ == "__main__":
    questions_path = "tests/fixtures/questions.jsonl"
    evaluate(questions_path)
