"""
llm_confusion_matrix.py

Tworzy macierz pomyłek metodą zero-shot przez lokalny LLM (gemma-3-4b w LM Studio,
endpoint http://localhost:1234/api/v1/chat) BEZ żadnego treningu.

Klasyfikuje dokładnie ten sam zbiór testowy co BERT (ten sam SEED i podział
z bert.py), prosząc model o przypisanie zgłoszenia do klasy 'support' lub
'technical', a następnie zapisuje wykres macierzy pomyłek.

Predykcje są zapisywane na bieżąco do models/llm/predictions.csv, więc po
przerwaniu skrypt wznawia pracę od miejsca, w którym skończył.
"""

import csv
import json
import sys
import time
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# ── Konfiguracja ──────────────────────────────────────────────────────────────

API_URL    = "http://localhost:1234/api/v1/chat"
MODEL      = "google/gemma-3-4b"
DATA_PATH  = Path("data/mapped.csv")
OUT_DIR    = Path("models/llm")
CM_PATH    = OUT_DIR / "confusion_matrix.png"
PRED_PATH  = OUT_DIR / "predictions.csv"

# Te same ustawienia podziału co w bert.py — żeby zbiór testowy był identyczny
TEST_SPLIT = 0.1
SEED       = 42

# Ograniczenie długości treści, żeby prompt nie był zbyt długi
BODY_CHARS = 1500
REQ_TIMEOUT = 120
MAX_RETRIES = 3

CLASSES = ["support", "technical"]
label2id = {"support": 0, "technical": 1}

SYSTEM_PROMPT = (
    "You are a support ticket classifier. "
    "Classify each ticket into exactly ONE category:\n"
    "- 'technical': technical problems, bugs, errors, connectivity, hardware/software malfunctions.\n"
    "- 'support': general customer service, billing, orders, returns, account or how-to questions.\n"
    "Answer with ONLY one single word: technical or support. No other text."
)

# ── 1. Wczytanie danych i odtworzenie zbioru testowego ────────────────────────

def load_data(path: Path):
    subjects, bodies, labels = [], [], []
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            subjects.append(row["subject"])
            bodies.append(row["body"])
            labels.append(row["category"])
    return subjects, bodies, labels

subjects, bodies, labels = load_data(DATA_PATH)
label_ids = [label2id[l] for l in labels]
indices = list(range(len(labels)))

# Pierwszy split z bert.py oddziela zbiór testowy — powtarzamy go na indeksach
_, test_idx = train_test_split(
    indices,
    test_size=TEST_SPLIT,
    random_state=SEED,
    stratify=label_ids,
)

# Opcjonalny limit liczby próbek: python llm_confusion_matrix.py 100
# (zbiór testowy jest już potasowany i zbalansowany, więc bierzemy pierwsze N)
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else None
if LIMIT:
    test_idx = test_idx[:LIMIT]
print(f"Zbiór testowy: {len(test_idx)} próbek")

# ── 2. Klient LLM ─────────────────────────────────────────────────────────────

def classify(subject: str, body: str) -> str:
    """Zwraca 'support' lub 'technical' (lub 'unknown' gdy nie da się sparsować)."""
    text = f"Subject: {subject}\n\nBody: {body[:BODY_CHARS]}"
    payload = json.dumps({
        "model": MODEL,
        "system_prompt": SYSTEM_PROMPT,
        "input": text,
    }).encode("utf-8")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(
                API_URL, data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=REQ_TIMEOUT) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            content = data["output"][0]["content"].strip().lower()
            # Parsowanie odpowiedzi
            if "technical" in content:
                return "technical"
            if "support" in content:
                return "support"
            return "unknown"
        except Exception as e:
            if attempt == MAX_RETRIES:
                print(f"  [błąd po {MAX_RETRIES} próbach] {e}")
                return "unknown"
            time.sleep(2 * attempt)

def main():
    # ── 3. Wznowienie z zapisanych predykcji ──────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    done = {}  # indeks zgłoszenia -> przewidziana etykieta
    if PRED_PATH.exists():
        with PRED_PATH.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                done[int(row["idx"])] = row["pred"]
        print(f"Wznowienie: wczytano {len(done)} wcześniejszych predykcji")

    # ── 4. Klasyfikacja całego zbioru testowego ───────────────────────────────
    write_header = not PRED_PATH.exists()
    total = len(test_idx)
    start = time.time()

    with PRED_PATH.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["idx", "true", "pred"])

        for n, i in enumerate(test_idx, 1):
            if i in done:
                continue
            pred = classify(subjects[i], bodies[i])
            done[i] = pred
            writer.writerow([i, labels[i], pred])
            f.flush()

            if n % 25 == 0 or n == total:
                elapsed = time.time() - start
                rate = n / elapsed if elapsed else 0
                print(f"  {n}/{total}  ({rate:.1f} zgłoszeń/s)")

    # ── 5. Zestawienie wyników ─────────────────────────────────────────────────
    y_true, y_pred = [], []
    unknown = 0
    for i in test_idx:
        pred = done[i]
        if pred == "unknown":
            unknown += 1
            continue
        y_true.append(label2id[labels[i]])
        y_pred.append(label2id[pred])

    if unknown:
        print(f"\nUWAGA: {unknown} odpowiedzi nie udało się sparsować (pominięte).")

    print("\n=== Raport klasyfikacji LLM (zbiór testowy) ===")
    print(classification_report(y_true, y_pred, target_names=CLASSES))

    # ── 6. Macierz pomyłek ─────────────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Greens",
        xticklabels=CLASSES, yticklabels=CLASSES,
    )
    plt.title(f"Macierz Pomyłek: LLM ({MODEL}, zero-shot)")
    plt.ylabel("Rzeczywista kategoria")
    plt.xlabel("Przewidziana kategoria")
    plt.savefig(CM_PATH, bbox_inches="tight")
    print(f"\nZapisano macierz pomyłek w: {CM_PATH}")


if __name__ == "__main__":
    main()
