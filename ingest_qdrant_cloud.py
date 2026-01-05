import os
import glob
from typing import List

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse

# ---------- Настройки ----------

COLLECTION_NAME = "moedelo_reforma_2026"

INPUT_DIR = "tokens"

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# ---------- Инициализация ----------

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

if not openai_api_key:
    raise RuntimeError("Не найден OPENAI_API_KEY в .env")
if not qdrant_url:
    raise RuntimeError("Не найден QDRANT_URL в .env")
if not qdrant_api_key:
    raise RuntimeError("Не найден QDRANT_API_KEY в .env")

client = OpenAI(api_key=openai_api_key)

qdrant = QdrantClient(
    url=qdrant_url,
    api_key=qdrant_api_key,
    timeout=30,
)

# ---------- Вспомогательные функции ----------

def embed_text(text: str) -> List[float]:
    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    return resp.data[0].embedding


def split_text(text: str) -> List[str]:
    text = " ".join(text.strip().split())
    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end]

        if end < len(text):
            last_space = chunk.rfind(" ")
            if last_space != -1:
                chunk = chunk[:last_space]
                end = start + last_space

        chunks.append(chunk)
        start = max(end - CHUNK_OVERLAP, end)

    return chunks


def ensure_collection():
    try:
        info = qdrant.get_collection(COLLECTION_NAME)
        size = None

        vectors = info.config.params.vectors
        if hasattr(vectors, "size"):
            size = vectors.size
        elif isinstance(vectors, dict):
            size = list(vectors.values())[0].size

        print(f"[✓] Коллекция '{COLLECTION_NAME}' существует, размер вектора = {size}")

        if size != EMBEDDING_DIM:
            print(f"[!] Несоответствие размерности: коллекция={size}, модель={EMBEDDING_DIM}")

    except UnexpectedResponse as e:
        if e.status_code == 404:
            print(f"[!] Коллекция '{COLLECTION_NAME}' отсутствует — создаю...")
            qdrant.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=qmodels.VectorParams(
                    size=EMBEDDING_DIM,
                    distance=qmodels.Distance.COSINE,
                ),
            )
            print("[+] Коллекция создана.")
        else:
            raise


# ---------- Основная логика ----------

def ingest_files():
    print(f"Загрузка данных в '{COLLECTION_NAME}'...")
    print(f"Папка: {INPUT_DIR}")

    ensure_collection()

    files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.txt")))
    print(f"Найдено файлов: {len(files)}")

    if not files:
        print("Нет файлов для обработки.")
        return

    vectors = []
    payloads = []
    ids = []
    pid = 1

    for idx, path in enumerate(files, start=1):
        print(f"[{idx}] Файл: {os.path.basename(path)}")

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = split_text(text)
        if not chunks:
            print("   Пусто — пропуск.")
            continue

        for ch in chunks:
            try:
                emb = embed_text(ch)
            except Exception as e:
                print(f"Ошибка эмбеддинга: {e}")
                continue

            vectors.append(emb)
            payloads.append({
                "title": os.path.basename(path),
                "content": ch,
            })
            ids.append(pid)
            pid += 1

    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=qmodels.Batch(
            ids=ids,
            vectors=vectors,
            payloads=payloads,
        ),
    )

    print(f"[✓] Загружено {len(vectors)} фрагментов в Qdrant Cloud.")


if __name__ == "__main__":
    ingest_files()
