import sys
import requests
from bs4 import BeautifulSoup
import json
import os
import time
from langdetect import detect, LangDetectException
from requests.exceptions import RequestException, Timeout
import random
import pdfplumber
from duckduckgo_search import DDGS
from concurrent.futures import ThreadPoolExecutor
import threading

sys.stdout.reconfigure(encoding='utf-8')

# =======================
# USER CONFIGURATION
# =======================
QUERIES = [
   "Project Management", "मशीन लर्निंग एल्गोरिदम", "Machine Learning Algorithms", 
   "स्मार्ट डिवाइस इंटरफेस", "Smart Device Interface", "आर्टिफिशियल इंटेलिजेंस आधारित", 
   "Artificial Intelligence-Based", "डेटा स्रोत", "Data Source",
   #Add as many queries as required
]

OUTPUT_DIR = "output_data"
URL_FILE = "urls.txt"
PDF_FILE_DIR = "pdf_files"
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PDF_FILE_DIR, exist_ok=True)

# Thread lock for safe file writing
file_lock = threading.Lock()

# =======================
# HELPER FUNCTIONS
# =======================
def contains_nepali(text):
    """Check if the text contains Nepali (Devanagari) characters."""
    return any('\u0900' <= ch <= '\u097F' for ch in text)

def is_eastern_religion(text):
    """Filter out texts mentioning non-Eastern religious terms."""
    exclude_keywords = ["Islam", "Muslim", "Middle East", "Islamic", "arabic", "arab", "persia", "terror", "terrorism"]
    return not any(kw.lower() in text.lower() for kw in exclude_keywords)

def is_valid_content(text):
    """Validate if content is relevant and in English/Nepali."""
    if not is_eastern_religion(text):
        return False
    try:
        lang = detect(text)
        return lang in ["en", "ne"]
    except LangDetectException:
        return False

def get_random_user_agent():
    """Return a random User-Agent string."""
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.5359.125 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.5304.110 Safari/537.36"
    ]
    return random.choice(user_agents)

def search_duckduckgo(query):
    """Search DuckDuckGo using API while avoiding rate limits."""
    results = []
    try:
        with DDGS() as ddgs:
            for result in ddgs.text(query, max_results=5):  # Reduce max results to 5
                if "href" in result:
                    results.append(result["href"])
        time.sleep(random.uniform(5, 10))  # Random delay to avoid rate limiting
    except Exception as e:
        print(f"Error in search: {e}")
    return results

def extract_content(url):
    """Scrape a webpage and extract relevant content."""
    try:
        headers = {"User-Agent": get_random_user_agent()}  # Use random User-Agent
        response = requests.get(url, timeout=15, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        title = soup.title.string.strip() if soup.title else "No Title"
        content = " ".join(p.get_text().strip() for p in soup.find_all("p") if p.get_text().strip())
        return title, content
    except (RequestException, Timeout):
        return None, None

def process_query(query):
    """Process a single query: search, extract, validate, and save."""
    urls = search_duckduckgo(query)

    with open(URL_FILE, "a", encoding="utf-8") as url_file:
        for url in urls:
            url_file.write(url + "\n")

            title, content = extract_content(url)
            if title and content and is_valid_content(title + " " + content):
                save_data({
                    "input": f"Summarize: {title}",
                    "target": content
                })
            time.sleep(random.uniform(2, 5))  # Small delay between requests

def get_next_available_file():
    """Find the next available file that does not exceed the max file size."""
    file_index = 1
    while True:
        current_file = os.path.join(OUTPUT_DIR, f"data_part_{file_index}.jsonl")
        if not os.path.exists(current_file) or os.path.getsize(current_file) < MAX_FILE_SIZE:
            return current_file
        file_index += 1

def save_data(data):
    """Save data in JSONL format, splitting files if needed."""
    with file_lock:
        current_file = get_next_available_file()
        with open(current_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF using pdfplumber."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages if page.extract_text())
            return text if text.strip() else None
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return None

def process_pdf(pdf_path):
    """Process a single PDF file: extract, validate, and save."""
    text = extract_text_from_pdf(pdf_path)
    if text and is_valid_content(text):
        save_data({
            "input": "Summarize the document",
            "target": text
        })

# =======================
# MAIN PROCESSING
# =======================
if __name__ == '__main__':
    # Process search queries
    with ThreadPoolExecutor(max_workers=3) as executor:  # Reduce threads to optimize performance
        executor.map(process_query, QUERIES)

    # Scan for PDFs in a directory and process them
    for pdf_file in os.listdir(PDF_FILE_DIR):
        if pdf_file.lower().endswith('.pdf'):
            process_pdf(os.path.join(PDF_FILE_DIR, pdf_file))

    print("Data collection and PDF processing completed!")
