from typing import List, Dict

from dotenv import dotenv_values

config = dotenv_values()

USE_LLM = int(config.get("USE_LLM", "1"))
MODEL_NAME = config.get("MODEL_NAME", "stub")


def generate_answer(query: str, retrieved: List[Dict]) -> str:
    """
    Генерация финального ответа на основе retrieved-контекста.
    Если USE_LLM=0 -> экстрактивный ответ (цитаты + краткая выжимка).
    Если USE_LLM=1 -> StubLLM перефразирует контекст.
    """
    if not retrieved:
        return "⚠️ Нет ответа (релевантность ниже порога)."

    if USE_LLM == 0:
        # Экстрактивный режим
        top_citations = retrieved[:3]
        parts = []
        for r in top_citations:
            parts.append(f"- {r['citation']}: {r['text']}")
        summary = "Найденные фрагменты:\n" + "\n".join(parts)
        return summary

    else:
        # Псевдо-LLM режим (перефразировка без выдумывания)
        context_texts = [r["text"] for r in retrieved[:3]]
        joined = " ".join(context_texts)
        return (
            f"💡 Сводка по запросу (модель={MODEL_NAME}):\n"
            f"{joined}\n\n"
            "Ответ составлен строго из найденных фрагментов."
        )
    