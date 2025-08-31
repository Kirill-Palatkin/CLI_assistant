import os
import re
import json
import pickle
from typing import List, Dict, Tuple

from dotenv import dotenv_values

config = dotenv_values()

chunk_size = int(config.get("CHUNK_SIZE", 150))
overlap = int(config.get("CHUNK_OVERLAP", 50))

WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def read_text_file(path: str) -> str:
    """Читает текстовый/markdown файл и возвращает строку."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def list_doc_files(docs_dir: str) -> List[str]:
    """Возвращает список текстовых файлов в директории (md/txt)."""
    exts = {".md", ".txt", ".markdown"}
    files = []
    for root, _, filenames in os.walk(docs_dir):
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() in exts:
                files.append(os.path.join(root, fn))
    return sorted(files)


def normalize_text(text: str) -> str:
    """Простейшая нормализация: убирает лишние пробелы, нормализует переносы."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{2,}", "\n\n", text)  # максимум 2 перевода строки подряд
    text = text.strip()
    return text


def tokenize_words(text: str) -> List[str]:
    """Токенизация на слова (простая)."""
    return WORD_RE.findall(text)


def chunk_text_by_words(text: str, chunk_size: int = 150, overlap: int = 50) -> List[Tuple[str,int,int]]:
    """
    Разбивает текст на чанки по словам с перекрытием.
    Возвращает список кортежей (chunk_text, start_word_idx, end_word_idx).
    """
    words = tokenize_words(text)
    if not words:
        return []
    chunks = []
    i = 0
    n = len(words)
    while i < n:
        end = min(i + chunk_size, n)
        chunk_words = words[i:end]
        chunk_text = " ".join(chunk_words)
        chunks.append((chunk_text, i, end-1))
        if end == n:
            break
        i = max(0, end - overlap)
    return chunks


def build_chunks_from_docs(docs_dir: str, chunk_size: int = 150, overlap: int = 50) -> List[Dict]:
    """
    Прочитать все файлы в docs_dir и вернуть список метаданных-чанков:
    each item: {
        'file': filepath,
        'chunk_id': int,
        'text': chunk_text,
        'start_word': int,
        'end_word': int
    }
    """
    files = list_doc_files(docs_dir)
    all_chunks = []
    cid = 0
    for fp in files:
        raw = read_text_file(fp)
        text = normalize_text(raw)
        chunks = chunk_text_by_words(text, chunk_size=chunk_size, overlap=overlap)
        for chunk_text, start, end in chunks:
            all_chunks.append({
                "file": os.path.relpath(fp),
                "chunk_id": cid,
                "text": chunk_text,
                "start_word": start,
                "end_word": end
            })
            cid += 1
    return all_chunks


def citation_tag(meta: Dict) -> str:
    """
    Формирует строку цитаты вида file:words[start-end]
    Вы можете улучшить до file:start_line-end_line при парсинге по строкам.
    """
    return f"{meta['file']}:words[{meta['start_word']}-{meta['end_word']}]"


def save_pickle(obj, path: str):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)


def to_pretty_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)
