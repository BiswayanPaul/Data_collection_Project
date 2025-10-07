import os
import re
import math
import time
import random
import logging
from typing import Dict, List, Set, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup
from newspaper import Article
import textstat
import nltk

# configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# Ensure required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords as nltk_stopwords

# requests session
SESSION = requests.Session()
# small set of user agents to rotate
USER_AGENTS = [
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.5938.92 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.5938.92 Safari/537.36'
]


def load_excel(file_path):
    return pd.read_excel(file_path)


def load_text(file_path):
    encodings = ("utf-8", "utf-8-sig", "latin-1")
    for enc in encodings:
        try:
            with open(file_path, 'r', encoding=enc) as file:
                return [line.rstrip('\n') for line in file]
        except UnicodeDecodeError:
            continue
    with open(file_path, 'r', encoding='latin-1', errors='replace') as file:
        return [line.rstrip('\n') for line in file]


def fetch_html(url):
    """Fetch a URL with retries, UA rotation and exponential backoff.

    Returns the page HTML on success, or None on repeated failure.
    """
    max_attempts = 3
    backoff_base = 1.5
    for attempt in range(1, max_attempts + 1):
        ua = random.choice(USER_AGENTS)
        headers = {
            'User-Agent': ua,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
        }
        try:
            resp = SESSION.get(url, timeout=15, headers=headers)
            resp.raise_for_status()
            return resp.text
        except requests.HTTPError as e:
            status = getattr(e.response, 'status_code', None)
            logging.warning('Attempt %d: HTTP %s for %s', attempt, status, url)
            # if 406/403, try again with a different UA after sleeping
            if attempt == max_attempts:
                logging.error('Failed fetching %s after %d attempts: %s', url, attempt, e)
                return None
        except requests.RequestException as e:
            logging.warning('Attempt %d: RequestException for %s: %s', attempt, url, e)
            if attempt == max_attempts:
                logging.error('Failed fetching %s after %d attempts: %s', url, attempt, e)
                return None
        # exponential backoff
        sleep_time = backoff_base ** attempt
        time.sleep(sleep_time)
    return None


def extract_article(html):
    """Attempt to extract title and article body using newspaper3k, fallback to BeautifulSoup heuristics."""
    # try newspaper Article if possible
    try:
        a = Article('')
        a.set_html(html)
        a.parse()
        title = a.title or ''
        body = a.text or ''
        if body.strip():
            return title, body
    except Exception:
        # fall through to soup heuristic
        pass

    soup = BeautifulSoup(html, 'html.parser')
    # title heuristic
    title_tag = soup.find('h1') or soup.find('title')
    title = title_tag.get_text(strip=True) if title_tag else ''

    # try article tag(s)
    article_tags = soup.find_all('article')
    if article_tags:
        parts = []
        for a in article_tags:
            parts.extend([p.get_text(strip=True) for p in a.find_all('p')])
        body = '\n'.join([p for p in parts if p])
        if body:
            return title, body

    # fallback: look for largest block
    candidates = soup.find_all(['div', 'section'], recursive=True)
    best = ''
    for c in candidates:
        text = c.get_text(separator=' ', strip=True)
        if len(text) > len(best):
            best = text
    if best:
        return title, best

    return title, soup.get_text(separator='\n', strip=True)


def clean_and_tokenize(text: str) -> List[str]:
    """Return a list of word tokens using NLTK's tokenizer.

    Keeps contractions and common word tokens.
    """
    # normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    tokens = word_tokenize(text)
    return tokens


def count_syllables(word):
    try:
        # textstat returns syllable count per word
        s = textstat.syllable_count(word)
        return max(1, s) if s is not None else 1
    except Exception:
        # fallback heuristic
        return max(1, len(re.findall(r'[aeiouyAEIOUY]+', word)))


def analyze_text(original_text: str, pos_set: Set[str], neg_set: Set[str], stopwords: Set[str]) -> Dict[str, float]:
    """Analyze text and compute the variables described in the assignment.

    Uses NLTK sentence and word tokenizers and textstat for syllable counting.
    """
    # sentences using NLTK
    sentences = sent_tokenize(original_text)
    sentences_count = max(1, len(sentences))

    tokens = clean_and_tokenize(original_text)
    # normalize tokens and filter stopwords and punctuation-only tokens
    words = [w for w in tokens if re.search('[A-Za-z0-9]', w) and w.lower() not in stopwords]
    word_count = len(words)

    # positive / negative
    pos_score = sum(1 for w in words if w.lower() in pos_set)
    neg_score = sum(1 for w in words if w.lower() in neg_set)

    # polarity and subjectivity
    polarity = (pos_score - neg_score) / ((pos_score + neg_score) + 1e-6)
    subjectivity = (pos_score + neg_score) / (word_count + 1e-6)

    # syllables and complex words
    syllables_total = 0
    complex_word_count = 0
    for w in words:
        s = count_syllables(w)
        syllables_total += s
        if s > 2:
            complex_word_count += 1

    syllable_per_word = (syllables_total / word_count) if word_count else 0.0

    avg_sentence_length = (word_count / sentences_count) if sentences_count else 0.0
    percentage_complex = (complex_word_count / word_count * 100) if word_count else 0.0
    fog_index = 0.4 * (avg_sentence_length + percentage_complex)

    avg_words_per_sentence = avg_sentence_length

    # personal pronouns (avoid matching 'US' as pronoun)
    pronoun_pattern = re.compile(r"\b(I|we|my|ours|us|our|me|you|your)\b", flags=re.I)
    pronouns = sum(1 for m in pronoun_pattern.finditer(original_text) if original_text[m.start():m.end()].upper() != 'US')

    avg_word_length = (sum(len(w) for w in words) / word_count) if word_count else 0.0

    return {
        'POSITIVE SCORE': pos_score,
        'NEGATIVE SCORE': neg_score,
        'POLARITY SCORE': polarity,
        'SUBJECTIVITY SCORE': subjectivity,
        'AVG SENTENCE LENGTH': avg_sentence_length,
        'PERCENTAGE OF COMPLEX WORDS': percentage_complex,
        'FOG INDEX': fog_index,
        'AVG NUMBER OF WORDS PER SENTENCE': avg_words_per_sentence,
        'COMPLEX WORD COUNT': complex_word_count,
        'WORD COUNT': word_count,
        'SYLLABLE PER WORD': syllable_per_word,
        'PERSONAL PRONOUNS': pronouns,
        'AVG WORD LENGTH': avg_word_length,
    }


def load_stopwords(stopwords_dir='StopWords'):
    sw = set()
    if not os.path.isdir(stopwords_dir):
        return sw
    for fname in os.listdir(stopwords_dir):
        path = os.path.join(stopwords_dir, fname)
        if os.path.isfile(path):
            lines = load_text(path)
            sw.update([l.strip().lower() for l in lines if l and not l.startswith('#')])
    return sw


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def main(test_mode=False, test_limit=1):
    df = load_excel('Input.xlsx')
    pos_set = set(w.lower() for w in load_text('MasterDictionary/positive-words.txt'))
    neg_set = set(w.lower() for w in load_text('MasterDictionary/negative-words.txt'))
    stopwords = load_stopwords('StopWords')
    # add NLTK stopwords (union) for robust removal
    stopwords = stopwords.union({s.lower() for s in nltk_stopwords.words('english')})

    ensure_dir('extracted_texts')

    results = []
    rows = df.to_dict('records')
    if test_mode:
        rows = rows[:test_limit]

    for row in rows:
        url = row.get('URL')
        url_id = str(row.get('URL_ID'))
        html = fetch_html(url)
        title = ''
        body = ''
        status = 'OK'
        error_msg = ''
        if html:
            try:
                title, body = extract_article(html)
            except Exception as e:
                title, body = '', ''
                status = 'ERROR'
                error_msg = str(e)
            # Save extracted text
            filename = os.path.join('extracted_texts', f"{url_id}.txt")
            try:
                with open(filename, 'w', encoding='utf-8') as fh:
                    fh.write(title + '\n' + body)
            except Exception as e:
                print(f"Error writing {filename}: {e}")
                status = 'ERROR'
                error_msg = str(e)
        else:
            status = 'FETCH_FAILED'
            error_msg = f'Failed to fetch {url}'
            print(f"Skipping analysis for {url} because fetch failed.")

        analysis = analyze_text((title + '\n' + body).strip(), pos_set, neg_set, stopwords)

        # combine row fields with analysis
        out_row = dict(row)
        out_row.update(analysis)
        out_row['STATUS'] = status
        out_row['ERROR'] = error_msg
        results.append(out_row)

    out_df = pd.DataFrame(results)

    # Ensure output columns follow the Output Data Structure.xlsx ordering
    expected_columns = [
        'URL_ID', 'URL', 'POSITIVE SCORE', 'NEGATIVE SCORE', 'POLARITY SCORE',
        'SUBJECTIVITY SCORE', 'AVG SENTENCE LENGTH', 'PERCENTAGE OF COMPLEX WORDS',
        'FOG INDEX', 'AVG NUMBER OF WORDS PER SENTENCE', 'COMPLEX WORD COUNT',
        'WORD COUNT', 'SYLLABLE PER WORD', 'PERSONAL PRONOUNS', 'AVG WORD LENGTH'
    ]

    # Start with the expected columns. If some are missing, add them with default values.
    for col in expected_columns:
        if col not in out_df.columns:
            out_df[col] = 0

    # Reorder accordingly (also keep any extra columns like STATUS/ERROR at the end)
    extra_cols = [c for c in out_df.columns if c not in expected_columns]
    ordered = expected_columns + extra_cols
    out_df = out_df[ordered]

    # Save files
    out_df.to_excel('output_analysis.xlsx', index=False)
    out_df.to_csv('output_analysis.csv', index=False)
    print(f"Saved analysis for {len(results)} articles to output_analysis.xlsx/csv")


if __name__ == '__main__':
    test_mode = os.environ.get('TEST_MODE') == '1'
    # if TEST_LIMIT env var present, use it
    test_limit = int(os.environ.get('TEST_LIMIT', '1')) if test_mode else 0
    main(test_mode=test_mode, test_limit=test_limit)
