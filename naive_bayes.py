"""
naive_bayes.py

Klasyfikacja zgłoszeń (technical / support) przy użyciu Naive Bayes.

Dane wejściowe: data/mapped.csv (wynik mapdata.py)
Wejście modelu: pole 'subject' + 'body'
Wyjście modelu: pole 'category' ('technical' lub 'support')
"""

import csv
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib

DATA_PATH  = Path("data/mapped.csv")
SAVE_PATH  = Path("models/naive_bayes.joblib")

TEST_SPLIT = 0.15
SEED       = 42

# ── 1. Wczytanie danych ───────────────────────────────────────────────────────

def load_data(path: Path) -> tuple[list[str], list[str]]:
    """Wczytuje CSV i zwraca (teksty, etykiety)."""
    texts, labels = [], []
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row["subject"] + " " + row["body"]
            texts.append(text)
            labels.append(row["category"])
    return texts, labels

texts, labels = load_data(DATA_PATH)
print(f"Wczytano {len(texts)} próbek.")
print(f"Rozkład klas: { {l: labels.count(l) for l in set(labels)} }")

# ── 2. Podział na train / test ────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels,
    test_size=TEST_SPLIT,
    random_state=SEED,
    stratify=labels,
)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# ── 3. Pipeline: TF-IDF + Naive Bayes ────────────────────────────────────────

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.90,
        sublinear_tf=True,
        strip_accents="unicode",
        token_pattern=r"(?u)\b[a-zA-ZÀ-ž]\w+\b",
    )),
    ("clf", ComplementNB(alpha=0.1)),
])

# ── 4. Trening ────────────────────────────────────────────────────────────────

print("\nTrenuję model...")
pipeline.fit(X_train, y_train)
print("Gotowe.")

# ── 5. Walidacja krzyżowa (5-fold CV) ────────────────────────────────────────

print("\nWalidacja krzyżowa (5-fold) na zbiorze treningowym...")
cv_scores = cross_val_score(
    pipeline, X_train, y_train,
    cv=5,
    scoring="f1_weighted",
    n_jobs=-1,
)
print(f"F1 per fold: {cv_scores.round(4)}")
print(f"Średnia F1:  {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ── 6. Ewaluacja na zbiorze testowym ─────────────────────────────────────────

y_pred = pipeline.predict(X_test)

print("\n=== Wyniki na zbiorze testowym ===")
print(classification_report(
    y_test, y_pred,
    digits=4,
))

# ── 7. Macierz pomyłek ────────────────────────────────────────────────────────

SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=sorted(set(labels)),
    colorbar=False,
    ax=ax,
)
ax.set_title("Macierz pomyłek – Naive Bayes")
plt.tight_layout()
plt.savefig("models/confusion_matrix.png", dpi=150)
plt.show()
print("Macierz pomyłek zapisana: models/confusion_matrix.png")

# ── 8. Najważniejsze słowa per klasa ─────────────────────────────────────────

print("\n=== Top 15 słów per klasa ===")
vocab = pipeline.named_steps["tfidf"].get_feature_names_out()
log_probs = pipeline.named_steps["clf"].feature_log_prob_
classes = pipeline.classes_

for i, cls in enumerate(classes):
    top_idx = log_probs[i].argsort()[-15:][::-1]
    top_words = vocab[top_idx]
    print(f"\n  {cls}: {', '.join(top_words)}")

# ── 9. Zapis modelu ───────────────────────────────────────────────────────────

joblib.dump(pipeline, SAVE_PATH)
print(f"\nModel zapisany w: {SAVE_PATH}")

# ── 10. Przykład inferencji ───────────────────────────────────────────────────

def predict(text: str) -> tuple[str, dict]:
    """Przewiduje kategorię i zwraca prawdopodobieństwa dla nowego tekstu."""
    proba = pipeline.predict_proba([text])[0]
    proba_dict = dict(zip(pipeline.classes_, proba.round(4)))
    pred = pipeline.predict([text])[0]
    return pred, proba_dict

example = "Server is down. Our production server stopped responding after the latest update."
pred, proba = predict(example)
print(f"\nPrzykład inferencji:")
print(f"  Tekst:              {example}")
print(f"  Predykcja:          {pred}")
print(f"  Prawdopodobieństwa: {proba}")
