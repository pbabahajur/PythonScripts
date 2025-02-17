import json
import re
import random
import asyncio
import logging
import os
from nltk.tokenize import sent_tokenize
from collections import Counter
from typing import List, Dict
import spacy

# ----------------------------
# Initialize models and tools
# ----------------------------
nlp = spacy.load("en_core_web_md")
npl = spacy.load("xx_ent_wiki_sm")

# ----------------------------
# Configurations and Constants
# ----------------------------
# Word count limits (adjust these as needed)
INPUT_LOWER = 4              # Lower bound for input text word count
TARGET_LOWER = 150            # Lower bound for target/value text word count (adjust as needed)
TARGET_UPPER = 400           # Upper bound for target/value text word count
BATCH_SIZE = 10              # Number of queries to process concurrently in a batch

# Maximum output file size (50 MB)
MAX_FILE_SIZE = 50 * 1024 * 1024

# Hard-coded file paths (input file and output folder)
INPUT_FILE_PATH = "output_data/data1.jsonl"
OUTPUT_DIR = "output-data"

# ----------------------------
# Pre-defined patterns/lists
# ----------------------------
UI_PHRASES = {
    "add to cart", "buy now", "click here", "read more", "sign up", "subscribe",
    "free trial", "learn more", "checkout", "order now", "view details", 
    "enable javascript", "please enable", "copyright", "all rights reserved", "All Rights Reserved"
}
SCRAPER_FAILURES = {"title", "heading", "null", "n/a", "loading...", "error", "undefined", "404"}

BOILERPLATE_PATTERNS = {
    'en': [
        r'bbc\.?com', r'\d{4} [A-Z]+', r'please enable javascript',
        r'external content', r'privacy policy', r'terms of use',
        r'© \d{4}', r'all rights reserved', r'published: \d{1,2} [A-Za-z]+ \d{4}',
        r'Google Scholar', r'doi:\s*\d+\.\d+', r'et al\.', r'pp\.\s*\d+'
    ],
    'ne': [
        r'प्रकाशित: \d{1,2} [^\s]+ \d{4}', r'स्रोत:', r'रिपब्लिक मिडिया',
        r'लेख्नुहोस्।', r'सहकार्य गरेर', r'अन्य वेबसाइट', r'गुगल स्कोलर'
    ]
}
ACRONYM_PATTERNS = {
    'en': {r'BSRS-5': 'Brief Symptom Rating Scale-5'},
    'ne': {r'स्वास्थ्य': 'स्वास्थ्य'}
}
forbidden_patterns = {
    'en': r'\b(?:save|download|pdf|email|phone|advertisement|sponsor)\b',
    'ne': r'(?:डाउनलोड|ईमेल|विज्ञापन|प्रायोजित|सदस्यता)'
}
# Range for Nepali Unicode characters (preserved as is)
NEPALI_UNICODE_RANGE = r'\u0900-\u097F\u200C-\u200D\u0964-\u0965'
NEPALI_STOP_WORDS = set([
    "छ", "गरेको", "गरी", "को", "र", "मा", "भन्ने", "ले", "का", "हो", "भएको",
    "तर", "यो", "त्यो", "पनि", "भने", "छन्", "गर्न", "हुन", "हुन्छ", "भनिए"
])
# ----------------------------
# Utility functions
# ----------------------------
def clean_jsonl_line(line: str) -> str:
    """Basic cleaning preserving structure."""
    cleaned = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', line)
    cleaned = cleaned.replace('�', "'")
    return cleaned.strip()

def repair_jsonl_line(line: str) -> str:
    """Conservative JSONL repair for common structural issues."""
    repairs = [
        (r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3'),   # Enclose unquoted keys
        (r':\s*([^"\d{][^,}]*)', r': "\1"'),       # Enclose unquoted string values
        (r'"\s*([^"]+?)"\s*([^"])', r'"\1", \2'),   # Insert missing commas
        (r'\\{', '{'), (r'\\}', '}')                # Fix escaped braces
    ]
    for pattern, repl in repairs:
        line = re.sub(pattern, repl, line)
    return line

def clean_text(text: str, lang: str) -> str:
    """
    Remove URLs, UI phrases, scraper failures, boilerplate and forbidden words;
    remove extra whitespaces and preserve Nepali Unicode.
    """
    text = re.sub(r'http\S+', '', text)
    for phrase in UI_PHRASES:
        text = re.sub(re.escape(phrase), '', text, flags=re.IGNORECASE)
    for phrase in SCRAPER_FAILURES:
        text = re.sub(re.escape(phrase), '', text, flags=re.IGNORECASE)
    for pattern in BOILERPLATE_PATTERNS.get(lang, []):
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    forbidden_pattern = forbidden_patterns.get(lang)
    if forbidden_pattern:
        text = re.sub(forbidden_pattern, '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def remove_redundant(text: str) -> str:
    """Remove redundant punctuation/characters."""
    text = re.sub(r'([!?.,]){2,}', r'\1', text)
    return text

def clean_generated_text(text: str) -> str:
    """Remove spurious special tokens (like <extra_id_0>) from the generated text."""
    return re.sub(r"<extra_id_\d+>", "", text).strip()

def count_capitalized_words(text: str) -> int:
    """Count words that start with a capital letter."""
    words = text.split()
    return sum(1 for word in words if word and word[0].isupper())

def split_text_into_segments(text: str, max_words: int, min_words: int) -> List[str]:
    """
    Split the text into segments each with at most `max_words` words.
    If the last segment is smaller than `min_words`, append it to the previous segment.
    """
    words = text.split()
    if len(words) <= max_words:
        return [text]
    segments = []
    i = 0
    while i < len(words):
        segment_words = words[i:i+max_words]
        if i + max_words < len(words):
            segments.append(" ".join(segment_words))
            i += max_words
        else:
            if segments and len(segment_words) < min_words:
                segments[-1] = segments[-1] + " " + " ".join(segment_words)
            else:
                segments.append(" ".join(segment_words))
            break
    return segments

def identify_relation(keywords):
    """
    Identify the key relationships among the keywords (for both Nepali and English).
    Returns lists of subjects, verbs, objects, locations, and organizations.
    """
    # Join the list of keywords into a sentence and parse with spaCy.
    doc = nlp(" ".join(keywords))
    
    subjects = [token for token in doc if token.dep_ in ('nsubj', 'nsubjpass')]
    verbs = [token for token in doc if token.pos_ == 'VERB']
    objects = [token for token in doc if token.dep_ in ('dobj', 'attr')]
    locations = [ent for ent in doc.ents if ent.label_ == 'GPE']  # Geopolitical entity (location)
    organizations = [ent for ent in doc.ents if ent.label_ == 'ORG']
    
    return subjects, verbs, objects, locations, organizations

def enforce_word_count(question: str, min_words: int = 4, max_words: int = 8) -> str:
    """
    Ensure the question has between min_words and max_words.
    If the question is longer than max_words, trim it.
    """
    words = question.split()
    if len(words) > max_words:
        words = words[:max_words]
        result = " ".join(words)
        if not result.endswith('?'):
            result += '?'
        return result
    return question

def generate_summary(text: str, lang: str, top_n: int = 100) -> list:
    """
    Extract the top N most frequent words from the given text.
    For English: use spaCy tokenization and lemmatization.
    For Nepali: split on whitespace and filter out stop words and words of length <= 3.
    """
    try:
        if lang == 'en':
            doc = nlp(text)
            tokens = [token.lemma_.lower() for token in doc if not token.is_stop and token.is_alpha]
        else:
            doc = npl(text)
            tokens = [word for word in text.split() if word not in NEPALI_STOP_WORDS and len(word) > 3]
        
        word_counts = Counter(tokens)
        return [word for word, _ in word_counts.most_common(top_n)]
    except Exception as e:
        logging.error(f"Word frequency extraction failed: {e}")
        return []

def generate_question(text: str, lang: str) -> str:
    """
    Generate a logical question from the given text.
    Supports both English and Nepali.
    Uses intelligent keyword relations to choose an appropriate question structure.
    Also writes the generated question to an output file.
    """
    if lang not in ['en', 'ne']:
        return "Question generation in the specified language is not supported yet."
    
    # Identify relationships using the keywords.
    
    question = ""
    if lang == 'en':
        keywords = generate_summary(text, lang)
        subjects, verbs, objects, locations, organizations = identify_relation(keywords)
        if locations:
            question_word = "Where"
            subject_str = locations[0].text
            context = [word for word in keywords if word.lower() != subject_str.lower()]
            question = f"{question_word} {subject_str} " + " ".join(context) + "?"
        elif subjects and verbs:
            subject_str = subjects[0].text
            context = [word for word in keywords if word.lower() != subject_str.lower()]
            verb_texts = [v.text.lower() for v in verbs]
            question_word = "What" if "training" in verb_texts else "How"
            question = f"{question_word} {subject_str} " + " ".join(context) + "?"
        else:
            question_word = "What"
            question = f"{question_word} is the relationship between " + ", ".join(keywords) + "?"
    
    else:  # Nepali branch
        keywords = generate_summary(text, lang)
        subjects, verbs, objects, locations, organizations = identify_relation(keywords)
        if locations:
            question_word = "कहाँ"
            subject_str = locations[0].text
            context = [word for word in keywords if word != subject_str]
            question = f"{question_word} {subject_str} " + " ".join(context) + "?"
        elif subjects and verbs:
            subject_str = subjects[0].text
            context = [word for word in keywords if word != subject_str]
            question = f"के {subject_str} " + " ".join(context) + " हुन्छ?"
        else:
            question_word = "के"
            question = f"{question_word} यी शब्दहरूको सम्बन्ध के हो: " + ", ".join(keywords) + "?"
    
    # Enforce that the question has between 4 and 8 words
    question = enforce_word_count(question, 4, 8)
    
    return question

async def process_single_query(query: Dict) -> List[Dict]:
    """
    Process one raw query dictionary and return a list of final query dicts.
    Each final dict is in the question-context-answer format.
    Returns an empty list if the query is filtered out.
    """
    results = []
    input_text = query.get('input', '')
    target_text = query.get('target', '') or query.get('value', '')
    
    lang = 'ne' if re.search(f'[{NEPALI_UNICODE_RANGE}]', target_text) else 'en'
    
    input_text = clean_text(input_text, lang)
    target_text = clean_text(target_text, lang)
    target_text = remove_redundant(target_text)
    
    words = target_text.split()
    if words:
        cap_count = count_capitalized_words(target_text)
        if (cap_count / len(words)) > 0.4:
            return []
    
    input_wc = len(input_text.split())
    target_wc = len(target_text.split())
    if target_wc < TARGET_LOWER:
        return []
    if input_wc < INPUT_LOWER and target_wc < TARGET_LOWER:
        return []
    
    segments = split_text_into_segments(target_text, TARGET_UPPER, TARGET_LOWER)
    
    for segment in segments:
        sentences = sent_tokenize(segment)
        if not sentences:
            continue
        
        # Define answer as all sentences in the segment
        answer = " ".join(sentences)
        # Generate context as a summary of the entire answer
        context = ' '.join(generate_summary(answer, lang))
        # Generate a high-quality question from the summarized context
        question = generate_question(answer, lang)
        
        if not (question and context and answer) or (question.strip() == context.strip() == answer.strip()):
            continue
        
        if not semantic_validation(question, context, answer):
            continue
        
        result = {
            "question": question,
            "context": context,
            "answer": answer
        }
        results.append(result)
    
    return results

def semantic_validation(question: str, context: str, answer: str) -> bool:
    """
    Placeholder semantic validation to ensure the three fields
    are not all identical. (Extend with proper checks as needed.)
    """
    return not (question.strip() == context.strip() == answer.strip())

async def process_batch(batch: List[Dict], write_lock: asyncio.Lock, output_writer, batch_number: int):
    """
    Process a batch of queries concurrently and then write each processed query
    in question-context-answer format to the output file (using a lock).
    """
    processed_queries = []
    tasks = [process_single_query(query) for query in batch]
    batch_results = await asyncio.gather(*tasks)
    for res in batch_results:
        if res:
            processed_queries.extend(res)
    
    # Remove duplicate queries (if question, context, and answer are all the same)
    unique_processed = []
    seen = set()
    for item in processed_queries:
        key = (item["question"].strip(), item["context"].strip(), item["answer"].strip())
        if key in seen:
            continue
        seen.add(key)
        unique_processed.append(item)
    
    async with write_lock:
        for item in unique_processed:
            output_writer.write(json.dumps(item, ensure_ascii=False) + "\n")
        output_writer.flush()
    logging.info(f"Processed batch {batch_number} and written {len(unique_processed)} queries.")

# ----------------------------
# Main event loop
# ----------------------------
async def main():
    logging.basicConfig(level=logging.INFO)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    batch = []
    batch_number = 1
    file_counter = 1
    current_output_file = os.path.join(OUTPUT_DIR, f"processed_data_{file_counter}.jsonl")
    output_writer = open(current_output_file, "w", encoding="utf-8")
    write_lock = asyncio.Lock()
    
    with open(INPUT_FILE_PATH, "r", encoding="utf-8") as infile:
        for line in infile:
            line = clean_jsonl_line(line)
            try:
                query = json.loads(line)
            except Exception as e:
                repaired_line = repair_jsonl_line(line)
                try:
                    query = json.loads(repaired_line)
                except Exception as e2:
                    logging.error(f"Skipping line due to JSON error: {e2}")
                    continue
            batch.append(query)
            
            if len(batch) >= BATCH_SIZE:
                await process_batch(batch, write_lock, output_writer, batch_number)
                batch_number += 1
                batch = []
                if os.path.getsize(current_output_file) >= MAX_FILE_SIZE:
                    output_writer.close()
                    file_counter += 1
                    current_output_file = os.path.join(OUTPUT_DIR, f"processed_data_{file_counter}.jsonl")
                    output_writer = open(current_output_file, "w", encoding="utf-8")
        if batch:
            await process_batch(batch, write_lock, output_writer, batch_number)
    output_writer.close()
    logging.info("All queries have been processed.")

if __name__ == "__main__":
    asyncio.run(main())