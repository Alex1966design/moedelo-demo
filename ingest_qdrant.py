import os
import uuid
from typing import List

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from openai import OpenAI

# -------------------------------------------------------------------
# Загрузка переменных окружения
# -------------------------------------------------------------------

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "moedelo_reforma_2026")

if not OPENAI_API_KEY:
    raise RuntimeError("Не найден OPENAI_API_KEY в .env")

# -------------------------------------------------------------------
# Инициализация клиентов
# -------------------------------------------------------------------

client = OpenAI(api_key=OPENAI_API_KEY)

qdrant = QdrantClient(
    host=QDRANT_HOST,
    port=QDRANT_PORT,
)


# -------------------------------------------------------------------
# Константы
# -------------------------------------------------------------------

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536  # размер вектора для text-embedding-3-small

# Мини-база знаний по реформе 2026 (примерные формулировки)
DOCUMENTS: List[str] = [
    "С 2026 года в РФ планируется комплексная реформа налоговой системы для малого и среднего бизнеса.",
    "Упрощённая система налогообложения (УСН) будет разделена на режимы с разным набором лимитов и требований к отчётности.",
    "Ставка НДС 20% сохраняется, но расширяются требования к электронному документообороту и прослеживаемости операций.",
    "Вводятся дополнительные критерии для признания компании высокотехнологичной и применения льготных ставок.",
    "Для самозанятых планируется повышение максимального лимита дохода и уточнение порядка взаимодействия с заказчиками – юрлицами.",
    "Часть налоговой отчётности будет переходить в полностью беззаявительный режим через цифровые сервисы ФНС.",
    "Ужесточаются требования к деловой цели операций и документальному подтверждению экономической обоснованности расходов.",
    "Планируется запуск цифровой платформы, которая автоматически рассчитывает налоговую нагрузку и напоминает о сроках уплаты.",
    "Для компаний на УСН изменяются лимиты по доходам и остаточной стоимости основных средств.",
    "Отдельные льготы будут привязаны к отраслевой принадлежности и уровню инвестиций в развитие бизнеса.",
]


# -------------------------------------------------------------------
# Вспомогательные функции
# -------------------------------------------------------------------

def get_embedding(text: str) -> List[float]:
    """
    Получить векторное представление текста через OpenAI embeddings.
    """
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    return response.data[0].embedding


def ensure_collection():
    """
    Создать коллекцию в Qdrant, если она ещё не создана.
    Если коллекция есть — просто используем её.
    """
    try:
        qdrant.get_collection(COLLECTION_NAME)
        print(f"Коллекция '{COLLECTION_NAME}' уже существует, переиспользуем.")
    except Exception:
        print(f"Коллекция '{COLLECTION_NAME}' не найдена, создаём новую...")
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=EMBEDDING_DIM,
                distance=models.Distance.COSINE,
            ),
        )
        print(f"Коллекция '{COLLECTION_NAME}' создана.")


def ingest_documents():
    """
    Основной пайплайн:
    - проверяем/создаём коллекцию;
    - считаем эмбеддинги для текстов;
    - upsert в Qdrant.
    """
    print("Инициализация коллекции Qdrant...")
    ensure_collection()

    print(f"Подготовлено документов: {len(DOCUMENTS)}")
    points: List[models.PointStruct] = []

    for idx, doc in enumerate(DOCUMENTS, start=1):
        print(f"[{idx}/{len(DOCUMENTS)}] Обработка документа...")
        vector = get_embedding(doc)

        point = models.PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "text": doc,
                "source": "moedelo_reforma_2026_demo",
                "index": idx,
            },
        )
        points.append(point)

    # upsert — вставка или обновление точек
    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )

    print(f"Загружено {len(points)} фрагментов в коллекцию '{COLLECTION_NAME}'.")
    print("Готово. Мини-база знаний по реформе 2026 загружена в Qdrant.")


# -------------------------------------------------------------------
# Точка входа
# -------------------------------------------------------------------

if __name__ == "__main__":
    ingest_documents()
