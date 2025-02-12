import sys
import os
import json
import time
import hashlib
import threading
import asyncio
import random
import io
import aiohttp
import pdfplumber
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from langdetect import detect, LangDetectException
import re

sys.stdout.reconfigure(encoding='utf-8')
SEED_TERMS = { 
    "abc": [
       "","","",
    ],
   "abc": [
       "","","",
    ],
   "abc": [
       "","","",
    ],
}

QUERIES = [
    f"{topic} for {context}, {fmt}"
    for topic in SEED_TERMS["abc"]
    for context in SEED_TERMS["abc"]
    for fmt in SEED_TERMS["abc"]
]

# =======================
# CONFIGURATION
# =======================


# Generating queries for both therapeutic techniques and mindfulness exercise


OUTPUT_DIR = "output_data"
URL_FILE = "processed_urls.txt"
PDF_DIR = "downloaded_pdfs"
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
MIN_CONTENT_LENGTH = 200  # Minimum content length to include

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PDF_DIR, exist_ok=True)

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36"
]

file_lock = threading.Lock()
processed_urls = set()
content_hashes = set()

# =======================
# CORE FUNCTIONS
# =======================
def get_random_user_agent():
    return random.choice(USER_AGENTS)

def load_processed_urls():
    if os.path.exists(URL_FILE):
        with open(URL_FILE, 'r', encoding='utf-8') as f:
            return set(line.strip() for line in f)
    return set()

def save_processed_url(url):
    with file_lock:
        with open(URL_FILE, 'a', encoding='utf-8') as f:
            f.write(url + '\n')
        processed_urls.add(url)

def get_output_file():
    index = 1
    while True:
        path = os.path.join(OUTPUT_DIR, f'data_{index}.jsonl')
        if not os.path.exists(path) or os.path.getsize(path) < MAX_FILE_SIZE:
            return path
        index += 1

def save_data(entries):
    if not entries:
        return

    with file_lock:
        output_file = get_output_file()
        with open(output_file, 'a', encoding='utf-8') as f:
            for entry in entries:
                entry_hash = hashlib.md5(json.dumps(entry, sort_keys=True).encode()).hexdigest()
                if entry_hash not in content_hashes:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                    content_hashes.add(entry_hash)

def is_valid_content(text):
    if len(text) < MIN_CONTENT_LENGTH:
        return False
    try:
        return detect(text) == 'en'
    except LangDetectException:
        return False

def extract_pairs(content):
    pairs = []
    
    # Try extracting heading-content patterns
    heading_re = r'(?:## |\b[A-Z][A-Za-z\s]+:|\d+\.\s+[A-Z][a-z]+)'
    sections = re.split(f'({heading_re})', content)
    
    current_heading = None
    for section in sections:
        section = section.strip()
        if re.match(heading_re, section):
            current_heading = section
        elif current_heading:
            if len(section) > len(current_heading) * 2:
                pairs.append((current_heading, section))
                current_heading = None
    
    # Fallback: split into paragraphs
    if not pairs:
        paragraphs = [p.strip() for p in content.split('\n') if p.strip()]
        for para in paragraphs:
            if len(para) > MIN_CONTENT_LENGTH:
                pairs.append((f"Explain: {para[:80]}...", para))
    
    return pairs

async def process_pdf(url, session):
    try:
        async with session.get(url) as response:
            content = await response.read()
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                text = "\n".join(page.extract_text() or "" for page in pdf.pages)
                if is_valid_content(text):
                    return extract_pairs(text)
    except Exception as e:
        print(f"Error processing PDF {url}: {e}")
    return []

async def process_html(url, session):
    try:
        async with session.get(url) as response:
            html = await response.text()
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'noscript']):
                element.decompose()
            
            # Extract main content
            text = soup.get_text(separator='\n')
            text = re.sub(r'\s+', ' ', text).strip()
            
            if is_valid_content(text):
                return extract_pairs(text)
    except Exception as e:
        print(f"Error processing HTML {url}: {e}")
    return []

async def search_and_process(query, session, semaphore):
    async with semaphore:
        print(f"Searching: {query}")
        
        try:
            with DDGS() as ddgs:
                results = [r['href'] for r in ddgs.text(query, max_results=5)]
        except Exception as e:
            print(f"Search error for {query}: {e}")
            return
        
        for url in results:
            if url in processed_urls:
                continue
            
            try:
                # Determine content type
                async with session.head(url) as response:
                    content_type = response.headers.get('Content-Type', '')
                    is_pdf = 'pdf' in content_type.lower()
                
                # Process content
                if is_pdf:
                    pairs = await process_pdf(url, session)
                else:
                    pairs = await process_html(url, session)
                
                # Prepare data entries
                entries = []
                for heading, content in pairs:
                    entries.append({
                        "input": heading,
                        "value": content
                    })
                
                if entries:
                    save_data(entries)
                    print(f"Saved {len(entries)} pairs from {url}")
                
                save_processed_url(url)
                time.sleep(random.uniform(1, 3))
                
            except Exception as e:
                print(f"Error processing {url}: {e}")

async def main():
    processed_urls.update(load_processed_urls())
    
    semaphore = asyncio.Semaphore(5)  # Concurrent requests
    timeout = aiohttp.ClientTimeout(total=20)
    
    async with aiohttp.ClientSession(headers={'User-Agent': get_random_user_agent()}, timeout=timeout) as session:
        tasks = []
        for query in QUERIES:
            task = asyncio.create_task(search_and_process(query, session, semaphore))
            tasks.append(task)
            time.sleep(0.5)  # Add delay between query starts
        
        await asyncio.gather(*tasks)

if __name__ == '__main__':
    asyncio.run(main())
