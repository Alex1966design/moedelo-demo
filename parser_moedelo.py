import os
import time
import urllib.parse
from typing import List, Tuple

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.moedelo.org/club/article-knowledge"
OUTPUT_DIR = "tokens"          # сюда будем сохранять статьи
SLEEP_BETWEEN_REQUESTS = 1.0   # пауза между запросами, чтобы не долбить сайт


def get_session() -> requests.Session:
    """
    Создаём сессию с нормальным User-Agent, чтобы сайт нас не резал.
    """
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0 Safari/537.36"
            )
        }
    )
    return session


def collect_article_links(session: requests.Session, limit: int = 30) -> List[Tuple[str, str]]:
    """
    Скачиваем страницу /club/article-knowledge и вытаскиваем ссылки на статьи.

    Подход:
      * берём все <a> с href, начинающимся с '/club/'
      * отбрасываем служебные страницы (авторы, теги, сама страница knowledge)
      * берём только те, у которых есть вменяемый текст-заголовок
      * ограничиваемся первыми `limit` уникальными ссылками
    """
    print(f"Скачиваю список статей с {BASE_URL} ...")
    resp = session.get(BASE_URL, timeout=20)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "lxml")

    links = {}
    for a in soup.find_all("a", href=True):
        href = a["href"]

        # интересуют только материалы клуба
        if not href.startswith("/club/"):
            continue

        # отсечём то, что почти точно не является отдельной статьёй
        if any(part in href for part in ("article-knowledge", "authors", "tag=")):
            continue

        url = urllib.parse.urljoin(BASE_URL, href)

        # заголовок статьи
        title = " ".join(a.stripped_strings)
        if not title or len(title) < 10:
            continue

        # убираем дубликаты
        if url in links:
            continue

        links[url] = title

        if len(links) >= limit:
            break

    result = list(links.items())
    print(f"Найдено статей: {len(result)}")
    return result


def extract_article_text(html: str) -> str:
    """
    Пытаемся вытащить основной текст статьи из HTML.

    Структура у «Моего дела» может меняться, поэтому делаем несколько попыток:
      * div с itemprop=articleBody
      * <article>
      * крупный div с классом, в имени которого есть 'article' или 'content'
    """
    soup = BeautifulSoup(html, "lxml")

    candidates = []

    # 1) по itemprop
    body = soup.find("div", attrs={"itemprop": "articleBody"})
    if body:
        candidates.append(body)

    # 2) тег <article>
    if not candidates:
        art = soup.find("article")
        if art:
            candidates.append(art)

    # 3) крупный див по классам
    if not candidates:
        for div in soup.find_all("div", class_=True):
            classes = " ".join(div.get("class", []))
            if any(key in classes.lower() for key in ("article", "content", "text")):
                # берём только достаточно "большие" блоки
                if len(div.get_text(strip=True)) > 500:
                    candidates.append(div)
                    break

    if not candidates:
        # fallback: весь текст страницы, но это крайний случай
        return soup.get_text(" ", strip=True)

    main = candidates[0]

    parts = []
    for tag in main.find_all(["h2", "h3", "p", "li"]):
        txt = tag.get_text(" ", strip=True)
        if txt:
            parts.append(txt)

    return "\n".join(parts)


def save_article(title: str, content: str, index: int) -> str:
    """
    Сохраняем статью в tokens/moedelo_article_XX.txt
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    safe_title = title.replace("\n", " ").strip()
    filename = f"moedelo_article_{index:02d}.txt"
    path = os.path.join(OUTPUT_DIR, filename)

    with open(path, "w", encoding="utf-8") as f:
        f.write(safe_title + "\n\n")
        f.write(content)

    return path


def main():
    session = get_session()

    articles = collect_article_links(session, limit=30)

    if not articles:
        print("Не удалось найти статьи. Проверь, не поменялась ли структура страницы.")
        return

    saved_paths = []

    for i, (url, title) in enumerate(articles, start=1):
        print(f"\n[{i}/{len(articles)}] Скачиваю статью:")
        print(f"  URL:   {url}")
        print(f"  Title: {title}")

        try:
            resp = session.get(url, timeout=20)
            resp.raise_for_status()
        except Exception as e:
            print(f"[!] Ошибка при запросе статьи: {e}")
            continue

        text = extract_article_text(resp.text)
        if not text or len(text) < 200:
            print("[!] Похоже, текст получился слишком коротким, пропускаю.")
            continue

        path = save_article(title, text, i)
        saved_paths.append(path)
        print(f"[+] Статья сохранена: {path}")

        time.sleep(SLEEP_BETWEEN_REQUESTS)

    print("\n=== Итог ===")
    print(f"Сохранено статей: {len(saved_paths)}")
    for p in saved_paths:
        print(" -", p)

    print("\nТеперь можно снова запустить ingest_qdrant.py, "
          "чтобы загрузить новые файлы в коллекцию 'moedelo_reforma_2026'.")


if __name__ == "__main__":
    main()
