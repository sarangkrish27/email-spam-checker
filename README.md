# 📧 Email Spam Classifier

A machine learning model that classifies emails as **spam** or **not spam**, built with scikit-learn and deployed as an interactive web app.

## Overview

This project trains a text classification pipeline on the [Spam Email Dataset](https://www.kaggle.com/datasets/jackksoncsie/spam-email-dataset) (via Kaggle) to detect spam emails, then wraps the trained model in a Streamlit app so anyone can paste in an email and get an instant prediction.

## Dataset

- **Source:** [jackksoncsie/spam-email-dataset](https://www.kaggle.com/datasets/jackksoncsie/spam-email-dataset) on Kaggle
- **Size:** 5,728 emails
- **Columns:** `text` (raw email content, including subject line) and `spam` (0 = not spam, 1 = spam)

## Approach

**1. Text cleaning**
Raw emails contain a `Subject:` prefix and long runs of underscores (`_ _ _ _ _`) used as visual separators. A custom `clean_text()` function strips both before the text reaches the vectorizer:

```python
def clean_text(txt):
    text = txt.replace("Subject:", "", 1).strip()
    text = re.sub(r'(?:_\s*){2,}', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
```

**2. Pipeline**
Cleaning, vectorization, and classification are chained into a single `sklearn.pipeline.Pipeline` so the exact same steps run at training time and prediction time:

```python
pipeline = Pipeline([
    ('clean', FunctionTransformer(clean_series)),
    ('tfidf', TfidfVectorizer(lowercase=True, stop_words='english')),
    ('model', LogisticRegression())
])
```

**3. Hyperparameter tuning**
`GridSearchCV` was used to search over TF-IDF n-gram range, minimum document frequency, and the model's regularization strength (`C`), optimizing for F1-score on the spam class.

```python
param_grid = {
    'tfidf__ngram_range': [(1,1), (1,2)],
    'tfidf__min_df': [1, 2],
    'model__C': [0.1, 1, 10]
}
```

## Results

| Metric (spam class) | Baseline | After tuning |
|---|---|---|
| Precision | 0.99 | 0.99 |
| Recall | 0.92 | 0.97 |
| F1-score | 0.95 | 0.98 |
| Overall accuracy | 0.98 | 0.99 |

Tuning reduced missed spam emails (false negatives) from 24 to 8 in the test set, while keeping false positives (real emails flagged as spam) at just 2.

**Confusion matrix (tuned model):**
```
[[854   2]
 [  8 282]]
```

## Project Structure

```
.
├── Email_Spam_Classification.ipynb   # training notebook: data loading, cleaning, pipeline, tuning
├── preprocessing.py                  # clean_text / clean_series, imported by both the notebook and the app
├── spam_pipeline.pkl                 # trained pipeline (cleaning + TF-IDF + Logistic Regression)
├── streamlit_app.py                  # web app for live predictions
└── README.md
```

## Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/email-spam-classifier.git
cd email-spam-classifier
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3a. Run the notebook (retrain from scratch)
Open `Email_Spam_Classification.ipynb` in Jupyter or Colab and run all cells. This downloads the dataset via `kagglehub`, retrains the pipeline, and regenerates `spam_pipeline.pkl`.

### 3b. Run the app (use the pre-trained model)
```bash
streamlit run streamlit_app.py
```
Paste any email text into the text box and click **Check email** to get a prediction with confidence score.

## Tech Stack

- **scikit-learn** — TF-IDF vectorization, Logistic Regression, pipeline, grid search
- **pandas** — data loading and manipulation
- **kagglehub** — dataset download
- **Streamlit** — web app interface
- **joblib** — model serialization

## Notes

- `preprocessing.py` must be imported the same way in both the notebook and the app — `joblib` needs to resolve `clean_text`/`clean_series` by their import path to unpickle the saved pipeline.
- The model was trained on a 2000s-era Enron/marketing email dataset; performance on modern email styles (e.g. phishing, promotional newsletters) is untested and may differ.