
import re
import httpx
import logging
from bs4 import BeautifulSoup
from readability import Document
from openai import OpenAI
from langchain.schema import Document as LCDocument
from langchain_core.runnables import Runnable
from langchain_core.tracers import tracing_v2_enabled
from playwright.async_api import async_playwright
import asyncio

from source_validator import is_reputable_source

# --- Logging setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hybrid_scraper")

# --- Step 1: Source validation ---
def validate_source(url: str) -> bool:
    valid = is_reputable_source(url)
    logger.info(f"Source validation for {url}: {'ACCEPTED' if valid else 'REJECTED'}")
    return valid

# --- Step 2: Check if the content is a professionally curated list ---
CURATED_LIST_PROMPT = """
You are a researcher assistant working on a restaurant discovery app. 
Your task is to decide if the given text is part of a professionally curated list of restaurants or cafes.
A professionally curated list is created by food critics, journalists, or reputable publications.
It usually includes a selection of venues with names, descriptions, and often reasons for recommendation.

Please answer "Yes" or "No" and explain briefly.
"""

def is_curated_list(openai_client: OpenAI, extracted_text: str) -> bool:
    logger.info("Checking if the article is a curated list...")
    response = openai_client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": CURATED_LIST_PROMPT},
            {"role": "user", "content": extracted_text[:3000]}
        ],
        temperature=0.0
    )
    answer = response.choices[0].message.content
    is_curated = "Yes" in answer
    logger.info(f"Curated list check: {'YES' if is_curated else 'NO'}\nReasoning: {answer.strip()}")
    return is_curated

# --- Step 3: Decide if we need Playwright ---
def needs_playwright(url: str, html: str) -> bool:
    dynamic = len(html) < 5000 or bool(re.search(r'<script[^>]*>.*?</script>', html, re.DOTALL))
    logger.info(f"Need Playwright for {url}: {'YES' if dynamic else 'NO'}")
    return dynamic

# --- Step 4: Fetch page content ---
async def fetch_with_playwright(url: str) -> str:
    logger.info(f"Fetching page with Playwright: {url}")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(url, timeout=60000)
        content = await page.content()
        await browser.close()
    return content

def fetch_with_httpx(url: str) -> str:
    logger.info(f"Fetching page with HTTPX: {url}")
    response = httpx.get(url, timeout=30)
    response.raise_for_status()
    return response.text

# --- Step 5: Extract relevant text blocks ---
def extract_text_blocks(html: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(['script', 'style', 'header', 'footer', 'nav', 'aside', 'form']):
        tag.decompose()
    text_blocks = []
    for tag in soup.find_all(['h1', 'h2', 'h3', 'p', 'li']):
        text = tag.get_text(strip=True)
        if text and len(text) > 30:
            text_blocks.append(text)
    logger.info(f"Extracted {len(text_blocks)} text blocks.")
    return text_blocks

def chunk_blocks(blocks: list[str], max_words=300) -> list[str]:
    chunks, current_chunk, word_count = [], [], 0
    for block in blocks:
        wc = len(block.split())
        if word_count + wc > max_words:
            chunks.append("\n".join(current_chunk))
            current_chunk, word_count = [], 0
        current_chunk.append(block)
        word_count += wc
    if current_chunk:
        chunks.append("\n".join(current_chunk))
    logger.info(f"Split content into {len(chunks)} chunks.")
    return chunks

# --- Step 6: Extract restaurant names + descriptions ---
EXTRACTION_PROMPT = """You are an expert travel journalist assistant. Your job is to extract the names of restaurants or cafés and their short descriptions from the following text. Output a list in this format:

Restaurant Name: <name>
Description: <short summary>

Only include actual food and beverage places. Do not include general tips, ads, or unrelated content."""

def extract_restaurants(chunk: str, openai_client: OpenAI) -> str:
    response = openai_client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": EXTRACTION_PROMPT},
            {"role": "user", "content": chunk}
        ],
        temperature=0.2
    )
    return response.choices[0].message.content

def deduplicate(results: list[str]) -> list[str]:
    seen, deduped = set(), []
    for item in results:
        name_match = re.search(r"Restaurant Name: (.+)", item)
        if name_match:
            name = name_match.group(1).strip().lower()
            if name not in seen:
                seen.add(name)
                deduped.append(item)
    logger.info(f"Deduplicated to {len(deduped)} unique restaurants.")
    return deduped

# --- LangChain Runnable ---
class HybridScraperRunnable(Runnable):
    def __init__(self, openai_client: OpenAI):
        self.openai_client = openai_client

    async def ainvoke(self, url: str) -> list[LCDocument]:
        with tracing_v2_enabled("hybrid_scraper"):
            if not validate_source(url):
                logger.warning(f"Rejected: Unreliable source -> {url}")
                return []

            raw_html = fetch_with_httpx(url)
            if needs_playwright(url, raw_html):
                raw_html = await fetch_with_playwright(url)

            doc = Document(raw_html)
            content_html = doc.summary()
            text_blocks = extract_text_blocks(content_html)

            if not is_curated_list(self.openai_client, "\n".join(text_blocks[:10])):
                logger.warning(f"Rejected: Not a curated list -> {url}")
                return []

            chunks = chunk_blocks(text_blocks)
            results = []
            for chunk in chunks:
                result = extract_restaurants(chunk, self.openai_client)
                results.append(result)

            deduped = deduplicate(results)

            logger.info(f"Finished scraping {url}")
            return [LCDocument(page_content=entry, metadata={"source": url}) for entry in deduped]
