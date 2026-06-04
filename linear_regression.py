"""
linear_regression.py

Klasyfikacja zgłoszeń (technical / support) przy użyciu regresji liniowej.

Regresja liniowa przewiduje wartość ciągłą, więc do klasyfikacji binarnej
kodujemy etykiety jako 0/1, dopasowujemy model na wektorach TF-IDF, a wynik
zamieniamy na klasę przez próg 0.5 (liniowy model prawdopodobieństwa).
Uwaga: dla klasyfikacji standardowym wyborem jest regresja logistyczna –
ten skrypt traktuje regresję liniową jako przykład poglądowy.

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
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib

DATA_PATH  = Path("data/mapped.csv")
SAVE_PATH  = Path("models/linear_regression.joblib")

TEST_SPLIT = 0.15
SEED       = 42
THRESHOLD  = 0.5

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

# ── 2. Kodowanie etykiet ──────────────────────────────────────────────────────

classes = sorted(set(labels))
label2id = {label: idx for idx, label in enumerate(classes)}
id2label = {idx: label for label, idx in label2id.items()}
label_ids = [label2id[l] for l in labels]
print(f"Klasy: {label2id}")

# ── 3. Podział na train / test ────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    texts, label_ids,
    test_size=TEST_SPLIT,
    random_state=SEED,
    stratify=label_ids,
)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# ── 4. Pipeline: TF-IDF + LinearRegression ───────────────────────────────────

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.90,
        sublinear_tf=True,
        strip_accents="unicode",
        token_pattern=r"(?u)\b[a-zA-ZÀ-ž]\w+\b",
    )),
    ("reg", LinearRegression()),
])

# ── 5. Trening ────────────────────────────────────────────────────────────────

print("\nTrenuję model...")
pipeline.fit(X_train, y_train)
print("Gotowe.")

# ── 6. Ewaluacja na zbiorze testowym ─────────────────────────────────────────

def predict_labels(texts: list[str]) -> list[int]:
    """Zamienia ciągłe wyjście regresji na klasy 0/1 progiem THRESHOLD."""
    scores = pipeline.predict(texts)
    return [int(s >= THRESHOLD) for s in scores]

y_pred = predict_labels(X_test)

print("\n=== Wyniki na zbiorze testowym ===")
print(classification_report(
    y_test, y_pred,
    target_names=classes,
    digits=4,
))

# ── 7. Macierz pomyłek ────────────────────────────────────────────────────────

SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=classes,
    colorbar=False,
    ax=ax,
)
ax.set_title("Macierz pomyłek – Regresja liniowa")
plt.tight_layout()
plt.savefig("models/confusion_matrix_linreg.png", dpi=150)
plt.show()
print("Macierz pomyłek zapisana: models/confusion_matrix_linreg.png")

# ── 8. Najważniejsze słowa per klasa ─────────────────────────────────────────

print("\n=== Top 15 słów wpływających na predykcję ===")
vocab = pipeline.named_steps["tfidf"].get_feature_names_out()
coef = pipeline.named_steps["reg"].coef_

order = coef.argsort()
top_pos = order[-15:][::-1]
top_neg = order[:15]

print(f"\n  W stronę '{id2label[1]}' (dodatnie wagi): {', '.join(vocab[top_pos])}")
print(f"\n  W stronę '{id2label[0]}' (ujemne wagi):  {', '.join(vocab[top_neg])}")

# ── 9. Zapis modelu ───────────────────────────────────────────────────────────

joblib.dump(pipeline, SAVE_PATH)
print(f"\nModel zapisany w: {SAVE_PATH}")

# ── 10. Przykład inferencji ───────────────────────────────────────────────────

def predict(text: str) -> tuple[str, float]:
    """Przewiduje kategorię i zwraca surowy wynik regresji dla nowego tekstu."""
    score = float(pipeline.predict([text])[0])
    pred = id2label[int(score >= THRESHOLD)]
    return pred, round(score, 4)

example = "Server is down. Our production server stopped responding after the latest update."
pred, score = predict(example)
print(f"\nPrzykład inferencji:")
print(f"  Tekst:      {example}")
print(f"  Predykcja:  {pred}")
print(f"  Wynik reg.: {score}  (próg = {THRESHOLD})")
