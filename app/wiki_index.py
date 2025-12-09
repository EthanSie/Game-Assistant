# app/wiki_index.py
import json
import math
from pathlib import Path
from typing import List, Dict
import requests
from bs4 import BeautifulSoup
import numpy as np
from urllib.parse import urljoin, urlparse

# Where we store the index
INDEX_PATH = Path("data/wiki_index.json")
INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

# Embedding model via Ollama
# app/wiki_index.py

EMBED_URL = "http://localhost:11434/api/embed"
EMBED_MODEL = "nomic-embed-text"

def embed_texts(texts: List[str], batch_size: int = 8) -> List[List[float]]:
    """
    Embed a list of texts using Ollama in small batches to avoid 500 errors.
    """
    all_embeddings: List[List[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        payload = {
            "model": EMBED_MODEL,
            "input": batch,
        }
        resp = requests.post(EMBED_URL, json=payload, timeout=300)
        try:
            resp.raise_for_status()
        except Exception as e:
            # If one batch fails, log it and skip that batch
            print(f"[embed_texts] Failed to embed batch {i}-{i+len(batch)}: {e}")
            continue

        data = resp.json()
        all_embeddings.extend(data["embeddings"])

    if len(all_embeddings) != len(texts):
        print(
            f"[embed_texts] Warning: only embedded {len(all_embeddings)} "
            f"out of {len(texts)} chunks."
        )

    return all_embeddings

def _index_document(url: str, title: str, text: str) -> Dict:
    chunks = chunk_text(text)
    if not chunks:
        raise ValueError("No text chunks extracted")

    embeddings = embed_texts(chunks)

    if not embeddings:
        raise ValueError("No embeddings returned for document")

    # If some batches failed, truncate to the min length so we don't mismatch
    n = min(len(chunks), len(embeddings))
    chunks = chunks[n - len(chunks):n] if len(chunks) != n else chunks
    embeddings = embeddings[:n]

    index = _load_index()
    doc_id = len(index["documents"])
    indexed_doc = {
        "id": doc_id,
        "url": url,
        "title": title,
        "chunks": [
            {"id": i, "text": chunks[i], "embedding": embeddings[i]}
            for i in range(n)
        ],
    }
    index["documents"].append(indexed_doc)
    _save_index(index)

    return {
        "id": doc_id,
        "url": url,
        "title": title,
        "num_chunks": n,
    }


# ---------- Index I/O ----------

def _load_index() -> Dict:
    if INDEX_PATH.exists():
        with open(INDEX_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"documents": []}

def _save_index(index: Dict):
    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False)

# ---------- Fetch, extract, chunk ----------

def fetch_html(url: str) -> str:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.text

def extract_text(html: str, url: str) -> Dict:
    soup = BeautifulSoup(html, "lxml")

    title = soup.title.string.strip() if soup.title and soup.title.string else url

    # Remove scripts/styles
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)

    return {"url": url, "title": title, "text": text}

def chunk_text(text: str, max_chars: int = 1500) -> List[str]:
    chunks = []
    current = []
    length = 0

    for paragraph in text.split("\n"):
        if not paragraph.strip():
            continue
        if length + len(paragraph) + 1 > max_chars and current:
            chunks.append("\n".join(current))
            current = []
            length = 0
        current.append(paragraph)
        length += len(paragraph) + 1

    if current:
        chunks.append("\n".join(current))

    return chunks

# ---------- Core indexing helpers ----------

def _index_document(url: str, title: str, text: str) -> Dict:
    chunks = chunk_text(text)
    if not chunks:
        raise ValueError("No text chunks extracted")

    embeddings = embed_texts(chunks)

    index = _load_index()
    doc_id = len(index["documents"])
    indexed_doc = {
        "id": doc_id,
        "url": url,
        "title": title,
        "chunks": [
            {"id": i, "text": chunk, "embedding": embeddings[i]}
            for i, chunk in enumerate(chunks)
        ],
    }
    index["documents"].append(indexed_doc)
    _save_index(index)

    return {
        "id": doc_id,
        "url": url,
        "title": title,
        "num_chunks": len(chunks),
    }

# Public: index a single URL
def index_url(url: str) -> Dict:
    html = fetch_html(url)
    doc = extract_text(html, url)
    return _index_document(doc["url"], doc["title"], doc["text"])

# ---------- Semantic search ----------

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def search_index(query: str, top_k: int = 5) -> List[Dict]:
    index = _load_index()
    if not index["documents"]:
        return []

    query_vec = np.array(embed_texts([query])[0], dtype="float32")

    scored = []
    for doc in index["documents"]:
        for chunk in doc["chunks"]:
            vec = np.array(chunk["embedding"], dtype="float32")
            sim = _cosine_sim(query_vec, vec)
            scored.append({
                "score": sim,
                "url": doc["url"],
                "title": doc["title"],
                "text": chunk["text"],
            })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]

# ---------- Crawler: crawl a site and index pages ----------

def _same_domain(url_a: str, url_b: str) -> bool:
    pa = urlparse(url_a)
    pb = urlparse(url_b)
    return pa.netloc == pb.netloc

def crawl_site(root_url: str, max_pages: int = 50) -> Dict:
    """
    Breadth-first crawl starting at root_url.
    Only follows links on the same domain.
    Indexes up to max_pages pages.
    Returns summary stats.
    """

    visited = set()
    queue = [root_url]
    indexed = 0

    while queue and indexed < max_pages:
        url = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)

        try:
            html = fetch_html(url)
        except Exception as e:
            print(f"[crawl] Failed to fetch {url}: {e}")
            continue

        # Extract and index the page text
        try:
            doc = extract_text(html, url)
            _index_document(doc["url"], doc["title"], doc["text"])
            indexed += 1
            print(f"[crawl] Indexed ({indexed}/{max_pages}): {url}")
        except Exception as e:
            print(f"[crawl] Failed to index {url}: {e}")

        # Find links to follow
        soup = BeautifulSoup(html, "lxml")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            # Build absolute URL
            next_url = urljoin(url, href)
            # Stay on the same domain
            if not _same_domain(root_url, next_url):
                continue
            # Skip fragments/mailto/etc.
            if next_url.startswith("mailto:") or "#" in next_url:
                continue
            if next_url not in visited and next_url not in queue:
                queue.append(next_url)

    return {
        "root_url": root_url,
        "pages_visited": len(visited),
        "pages_indexed": indexed,
        "max_pages": max_pages,
    }
