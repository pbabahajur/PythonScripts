import jsonlines
import re
import random
import string
import asyncio
import time
import json
import logging
from langdetect import detect, LangDetectException
from nltk.tokenize import sent_tokenize
from collections import Counter
from functools import partial
from typing import List, Dict
import aiohttp
import nltk

nltk.download('punkt')

logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s: %(message)s',
                    filename='preprocess.log',
                    filemode='w')
logger = logging.getLogger(__name__)

UI_PHRASES = {
    "add to cart", "buy now", "click here", "read more", "sign up", "subscribe",
    "free trial", "learn more", "checkout", "order now", "view details"
}
SCRAPER_FAILURES = {"title", "heading", "null", "n/a", "loading...", "error", "undefined"}

LOWER_WORD_THRESHOLD = 5
UPPER_WORD_THRESHOLD = 400

def remove_urls(text: str) -> str:
    return re.sub(r'http[s]?://\S+', '', text)

def clean_text(text: str) -> str:
    text = remove_urls(text)
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces/newlines with a single space
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def clean_special_characters(text: str) -> str:
    return re.sub(r'[^\x00-\x7F\u0900-\u097F\u0A00-\u0A7F\u0980-\u09FF\u0B80-\u0BFF\u0C80-\u0CFF\u2000-\u206F]+', '', text)

def remove_repeated_words(text: str) -> str:
    return re.sub(r'\b(\w+)\s+\1\b', r'\1', text)

def remove_leading_numbers(text: str) -> str:
    return re.sub(r'^\d+\s*', '', text)

def segment_sentences(text: str) -> List[str]:
    sentences = sent_tokenize(text)
    language_grouped_sentences = []
    temp_group = []
    current_language = None
    for sentence in sentences:
        try:
            detected_lang = detect(sentence)
            if current_language is None:
                current_language = detected_lang
            elif detected_lang != current_language:
                language_grouped_sentences.append(temp_group)
                temp_group = [sentence]
                current_language = detected_lang
            else:
                temp_group.append(sentence)
        except LangDetectException:
            continue
    if temp_group:
        language_grouped_sentences.append(temp_group)
    return language_grouped_sentences

def remove_noise(sentences: List[str]) -> List[str]:
    clean_sentences = []
    for sentence in sentences:
        if len(sentence.split()) > 2 and len(set(sentence)) > 3:
            clean_sentences.append(sentence)
    return clean_sentences

def deduplicate(texts: List[str]) -> List[str]:
    return list(set(texts))

def generate_context(target: str) -> str:
    templates = [
        f"This is a definition: {target}",
        f"The following statement is true: {target}",
        f"Here is some information: {target}",
        f"In this context, {target} is the answer.",
        f"The explanation for this is: {target}"
    ]
    return random.choice(templates)

def truncate_context(context: str, answer: str, threshold: float = 0.5) -> str:
    context_words = context.split()
    answer_words = answer.split()
    if not context_words:
        return context
    common = sum(1 for word in context_words if word.lower() in {w.lower() for w in answer_words})
    if common / len(context_words) > threshold and len(context_words) > 6:
        return ' '.join(context_words[:3] + context_words[-3:])
    return context

def is_target_double_size(input_text: str, target_value: str) -> bool:
    return len(target_value) >= 2 * len(input_text)

def has_minimum_word_count(text: str, min_words: int = LOWER_WORD_THRESHOLD) -> bool:
    return len(text.split()) >= min_words

def is_valid_question(text: str) -> bool:
    return text.strip().endswith('?')

def is_overly_redundant(input_text: str, answer: str, threshold: float = 0.6) -> bool:
    input_words = set(input_text.lower().split())
    answer_words = answer.lower().split()
    if not answer_words:
        return True
    common = sum(1 for word in answer_words if word in input_words)
    ratio = common / len(answer_words)
    return ratio >= threshold

def should_include_answer(answer: str) -> bool:
    if answer.strip().endswith('.'):
        return True
    word_count = len(answer.split())
    unwanted_phrases = UI_PHRASES.union(SCRAPER_FAILURES)
    if word_count > 200 and not any(phrase in answer.lower() for phrase in unwanted_phrases):
        return True
    return False

def has_forbidden_words(answer: str) -> bool:
    forbidden_words = ['save', 'pdf', 'download','disclaimer','copyright','email'] #add as many as you like
    word_count = Counter(re.findall(r'\b\w+\b', answer.lower()))
    return sum(word_count[word] for word in forbidden_words) >= 2

def load_json_data(filename: str) -> List[dict]:
    with jsonlines.open(filename) as f:
        return [line for line in f]

def save_incrementally_to_json(data, filename="qa_dataset.json"):
    with open(filename, "a", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        f.write("\n")

async def process_data_chunk(data_chunk: List[dict]) -> List[dict]:
    processed_data = []
    for entry in data_chunk:
        if 'input' not in entry or not (entry.get('value') or entry.get('target')):
            continue
        
        input_value = entry['input']
        target_value = entry.get('value', entry.get('target'))
        
        # Remove line breaks and clean text
        input_value = input_value.replace('\n', ' ').replace('\r', ' ')
        target_value = target_value.replace('\n', ' ').replace('\r', ' ')
        
        cleaned_input = clean_text(input_value)
        cleaned_input = clean_special_characters(cleaned_input)
        cleaned_input = remove_repeated_words(cleaned_input)
        cleaned_input = remove_leading_numbers(cleaned_input)
        
        cleaned_target = clean_text(target_value)
        cleaned_target = clean_special_characters(cleaned_target)
        cleaned_target = remove_repeated_words(cleaned_target)
        cleaned_target = remove_leading_numbers(cleaned_target)
        
        if not cleaned_target.strip():
            continue

        target_words = len(cleaned_target.split())
        processed = False
        
        if (target_words > 200 and 
            should_include_answer(cleaned_target) and 
            not has_forbidden_words(cleaned_target) and 
            not any(phrase in cleaned_target.lower() for phrase in UI_PHRASES)):
            
            input_word_count = len(cleaned_input.split())
            if input_word_count < LOWER_WORD_THRESHOLD:
                additional = ' '.join(cleaned_target.split()[:5])
                cleaned_input = f"What is {cleaned_input} {additional}?"
            
            if not is_valid_question(cleaned_input):
                cleaned_input = f"What is {cleaned_input}?"
            
            if not is_overly_redundant(cleaned_input, cleaned_target):
                context = generate_context(cleaned_target)
                context = truncate_context(context, cleaned_target)
                processed_data.append({
                    "question": cleaned_input,
                    "context": context,
                    "answer": cleaned_target
                })
                processed = True
        
        if not processed:
            if (has_minimum_word_count(cleaned_input) and
                is_target_double_size(cleaned_input, cleaned_target) and
                should_include_answer(cleaned_target) and
                not has_forbidden_words(cleaned_target) and
                not is_overly_redundant(cleaned_input, cleaned_target)):
                
                if not is_valid_question(cleaned_input):
                    if has_minimum_word_count(cleaned_input, LOWER_WORD_THRESHOLD + 2):
                        cleaned_input = f"What is {cleaned_input}?"
                    else:
                        additional = ' '.join(cleaned_target.split()[:5])
                        cleaned_input = f"What is {cleaned_input} {additional}?"
                
                if is_valid_question(cleaned_input):
                    context = generate_context(cleaned_target)
                    context = truncate_context(context, cleaned_target)
                    processed_data.append({
                        "question": cleaned_input,
                        "context": context,
                        "answer": cleaned_target
                    })
    
    return processed_data

async def process_data_in_parallel(data: List[dict], chunk_size: int = 50) -> List[dict]:
    tasks = []
    for i in range(0, len(data), chunk_size):
        tasks.append(process_data_chunk(data[i:i + chunk_size]))
    results = await asyncio.gather(*tasks)
    return [item for sublist in results for item in sublist]

async def preprocess_data(filename="abc"):
    data = load_json_data(filename)
    processed_data = await process_data_in_parallel(data)
    save_incrementally_to_json(processed_data)

if __name__ == "__main__":
    start_time = time.time()
    asyncio.run(preprocess_data("abc"))
    end_time = time.time()
    logger.info(f"Data preprocessing completed in {end_time - start_time:.2f} seconds.")