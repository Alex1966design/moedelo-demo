import os
from typing import Optional, Tuple

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient

# ==========================
#   ИНИЦИАЛИЗАЦИЯ
# ==========================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Не найден OPENAI_API_KEY в .env – добавь ключ в .env файл.")

# адрес локального Qdrant (как в ingest_qdrant.py)
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "moedelo_reforma_2026"

client = OpenAI(api_key=OPENAI_API_KEY)

qdrant = QdrantClient(
    url=QDRANT_URL,
    timeout=10.0  # чуть поменьше, чтобы не зависать
)

# ==========================
#   СИСТЕМНЫЙ ПРОМПТ LLM
# ==========================

SYSTEM_PROMPT = (
    "Ты внутренний AI-ассистент компании «Моё дело» для предпринимателей и бухгалтеров.\n\n"
    "Твоя задача — помогать разбираться именно в налоговой реформе 2026 года.\n"
    "Отвечай:\n"
    "• кратко и по делу,\n"
    "• простым, понятным языком без лишней бюрократии,\n"
    "• строго опираясь на переданный контекст (фрагменты базы знаний).\n\n"
    "Если в контексте нет достаточной информации для точного ответа:\n"
    "• честно скажи, что данных недостаточно,\n"
    "• НЕ выдумывай нормы законодательства и конкретные цифры,\n"
    "• можешь предложить сформулировать вопрос точнее.\n\n"
    "Не давай общих рассуждений вне темы реформы 2026 года и не используй контекст, "
    "который тебе явно не передан."
)

# ==========================
#   ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==========================

def embed_text(text: str) -> list:
    """
    Получаем эмбеддинг для текста с помощью OpenAI.
    Модель должна быть той же, что в ingest_qdrant.py.
    """
    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return emb.data[0].embedding


def search_qdrant(query_text: str, top_k: int = 5) -> Tuple[str, str]:
    """
    Ищем релевантные фрагменты в Qdrant и собираем контекст.
    Возвращаем: (context, debug_info)
    """
    query_vec = embed_text(query_text)

    res = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vec,
        limit=top_k,
        with_payload=True
    )

    if not res.points:
        return "", "Ничего не найдено в коллекции."

    context_chunks = []
    debug_lines = []

    for idx, p in enumerate(res.points, start=1):
        payload = p.payload or {}
        title = payload.get("title", "")
        content = payload.get("content", "")
        score = p.score

        piece = f"[{idx}] {title}\n{content}"
        context_chunks.append(piece)

        debug_lines.append(f"{idx}) score={score:.3f}, title={title}")

    context = "\n\n---\n\n".join(context_chunks)
    debug_info = "Найденные фрагменты:\n" + "\n".join(debug_lines)

    return context, debug_info


def ask_llm(question: str, context: str) -> str:
    """
    Отправляем запрос в LLM с учётом найденного контекста.
    """

    if context:
        user_content = (
            f"Вопрос пользователя:\n{question}\n\n"
            f"Контекст по реформе 2026 (фрагменты базы знаний):\n{context}"
        )
    else:
        user_content = (
            f"Вопрос пользователя:\n{question}\n\n"
            "Внимание: контекст из базы знаний не найден. "
            "Если ты не уверен, ответь, что информации недостаточно для точного ответа."
        )

    chat = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.3,
    )

    return chat.choices[0].message.content.strip()


def transcribe_audio(audio_path: str) -> str:
    """
    Пробуем распознать аудио через OpenAI.
    Если что-то пойдёт не так – кидаем исключение, а выше поймаем.
    """
    with open(audio_path, "rb") as f:
        result = client.audio.transcriptions.create(
            model="gpt-4o-transcribe",
            file=f,
        )
    return result.text.strip()


# ==========================
#   ЛОГИКА ОБРАБОТКИ ЗАПРОСА
# ==========================

def handle_query(text_question: str, audio_file: Optional[str]) -> str:
    try:
        # 1. Получаем вопрос: текстом или голосом
        question = (text_question or "").strip()

        if not question and audio_file:
            try:
                question = transcribe_audio(audio_file)
            except Exception as e:
                return (
                    "Не удалось распознать аудио. "
                    f"Ошибка: {e}\n\nПопробуйте задать вопрос текстом."
                )

        if not question:
            return "Пожалуйста, задайте вопрос текстом или голосом 🙂"

        # 2. Ищем релевантные фрагменты в Qdrant
        context, debug_info = search_qdrant(question)

        # 3. Спрашиваем LLM
        answer = ask_llm(question, context)

        # 4. Формируем красивый ответ
        parts = [f"**Вопрос:** {question}\n", f"**Ответ ассистента:**\n{answer}"]

        if context:
            parts.append("\n---\n**Использованный контекст из базы знаний:**\n")
            parts.append(context)

        # отладочная информация, можно убрать, если мешает
        parts.append("\n---\n<details><summary>Отладочная информация (для демо)</summary>\n\n")
        parts.append(debug_info)
        parts.append("\n</details>")

        return "\n".join(parts)

    except Exception as e:
        # Любая непойманная ошибка – показываем текст + печатаем трейсбек в консоль
        import traceback
        traceback.print_exc()
        return (
            "⚠️ В демо-версии ассистента произошла внутренняя ошибка.\n"
            f"`{e}`\n\n"
            "Проверьте, запущен ли Qdrant и корректен ли OPENAI_API_KEY."
        )


# ==========================
#   GRADIO UI
# ==========================

def create_demo():
    with gr.Blocks(title="Моё дело — Реформа 2026") as demo:
        gr.Markdown(
            """
            # 🤖 AI-ассистент «Моё дело» по реформе 2026

            Задайте вопрос **текстом или голосом** — ассистент ответит, опираясь на мини-базу знаний,
            собранную из материалов о налоговой реформе 2026 года.
            """
        )

        with gr.Row():
            text_in = gr.Textbox(
                label="Вопрос текстом",
                placeholder="Например: Какие изменения будут в налоговом законодательстве РФ в 2026 году?",
            )

        audio_in = gr.Audio(
            sources=["microphone", "upload"],
            type="filepath",
            label="Или задайте вопрос голосом (микрофон или аудиофайл)",
        )

        ask_btn = gr.Button("Спросить ассистента", variant="primary")
        output_md = gr.Markdown(label="Ответ")

        ask_btn.click(
            fn=handle_query,
            inputs=[text_in, audio_in],
            outputs=output_md,
        )

        return demo


demo_app = create_demo()

if __name__ == "__main__":
    # запустим на localhost:7860 — как ты уже привык
    demo_app.launch(share=True)

