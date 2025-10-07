# Data Extraction & NLP Assignment

This project extracts article text from URLs in `Input.xlsx`, computes text analysis variables (sentiment, readability, etc.), and writes results to `output_analysis.xlsx` and `output_analysis.csv`.

Requirements:

- Python 3.10+ (virtualenv recommended)
- Install dependencies in `requirements.txt` into a virtual environment.

Quick start:

```bash
# activate your existing venv
source .venv/bin/activate

# (optional) install requirements if you changed them
python -m pip install -r requirements.txt

# run in test mode to process only first N rows
TEST_MODE=1 TEST_LIMIT=5 python3 src/main.py

# run full dataset
python3 src/main.py
```

Outputs:

- `extracted_texts/<URL_ID>.txt` — extracted title + body
- `output_analysis.xlsx` and `output_analysis.csv` — final analysis results

Notes:

- The script uses `newspaper3k` for extraction with a BeautifulSoup fallback.
- If a site blocks requests (HTTP 403/406), the script records `STATUS=FETCH_FAILED` and continues.
- For dynamic or heavily protected sites you may need to use Selenium or provide cookies.

If you want, I can add a CLI, unit tests, and run the full dataset and share the resulting files.
