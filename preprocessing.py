import re

def clean_text(txt):
    text = txt.replace("Subject:", "", 1).strip()
    text = re.sub(r'(?:_\s*){2,}', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_series(X):
    return [clean_text(t) for t in X]