# app/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import HTMLResponse

from .llm_client import call_llm
from .wiki_index import index_url, search_index, crawl_site

app = FastAPI()

class AskRequest(BaseModel):
    message: str

class AskResponse(BaseModel):
    answer: str

class IndexUrlRequest(BaseModel):
    url: str

class IndexUrlResponse(BaseModel):
    id: int
    url: str
    title: str
    num_chunks: int

class CrawlRequest(BaseModel):
    root_url: str
    max_pages: int = 50

class CrawlResponse(BaseModel):
    root_url: str
    pages_visited: int
    pages_indexed: int
    max_pages: int

BASE_SYSTEM_PROMPT = """You are a helpful game assistant.
You may be given reference passages from previously indexed web pages.
Use those passages as the primary source when they are present.
"""

@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <html>
      <body>
        <h1>Game Assistant API</h1>
        <p>POST /ask {"message": "your question"}</p>
        <p>POST /index_url {"url": "https://example.com/page"}</p>
        <p>POST /crawl_site {"root_url": "https://example.com", "max_pages": 50}</p>
      </body>
    </html>
    """

@app.post("/index_url", response_model=IndexUrlResponse)
def index_url_endpoint(req: IndexUrlRequest):
    result = index_url(req.url)
    return IndexUrlResponse(**result)

@app.post("/crawl_site", response_model=CrawlResponse)
def crawl_site_endpoint(req: CrawlRequest):
    result = crawl_site(req.root_url, max_pages=req.max_pages)
    return CrawlResponse(**result)

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    user_message = req.message
    lower = user_message.lower()
    context_prefix = ""

    # For now, assume anything mentioning Minecraft / craft / recipe
    # should use indexed wiki passages.
    if "minecraft" in lower or "craft" in lower or "recipe" in lower:
        matches = search_index(user_message, top_k=5)
        if matches:
            context_prefix += "Here are some relevant reference passages:\n\n"
            for m in matches:
                context_prefix += f"From {m['title']} ({m['url']}):\n{m['text']}\n\n"

    full_user_prompt = context_prefix + user_message
    answer = call_llm(BASE_SYSTEM_PROMPT, full_user_prompt)
    return AskResponse(answer=answer)
