import pytest
from src.main import analyze_text


def test_analyze_basic():
    text = "I love this product. It is great and amazing."
    pos_set = {"love", "great", "amazing"}
    neg_set = {"bad", "terrible"}
    stopwords = set()

    out = analyze_text(text, pos_set, neg_set, stopwords)
    assert out['POSITIVE SCORE'] == 3
    assert out['NEGATIVE SCORE'] == 0
    assert out['WORD COUNT'] > 0
    assert 'POLARITY SCORE' in out
