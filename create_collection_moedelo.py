import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Загружаем переменные из .env
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Имя коллекции должно совпадать с тем, что в app.py
# В .env можно явно задать QDRANT_COLLECTION_NAME, иначе возьмём дефолт.
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "moedelo_reforma_2026")


def main():
    if not QDRANT_URL:
        raise RuntimeError("QDRANT_URL не задан в .env")
    if not QDRANT_API_KEY:
        raise RuntimeError("QDRANT_API_KEY не задан в .env")

    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=20.0,
    )

    # text-embedding-3-small даёт эмбеддинги размерности 1536
    # (важно совпадение с моделью в app.py)
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=1536,
            distance=Distance.COSINE,
        ),
    )

    print(f"Коллекция '{COLLECTION_NAME}' создана/пересоздана.")


if __name__ == "__main__":
    main()
