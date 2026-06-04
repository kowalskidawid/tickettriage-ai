"""
bert_confusion_matrix.py

Tworzy macierz pomyłek dla WYTRENOWANEGO modelu BERT z models/bert_category
BEZ ponownego treningu.

Odtwarza dokładnie ten sam zbiór testowy co w bert.py (ten sam SEED, ten sam
podział i stratyfikacja), wczytuje zapisany model, wykonuje predykcję i zapisuje
wykres macierzy pomyłek do models/bert_category/confusion_matrix.png
"""

import csv
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# ── Te same ustawienia co w bert.py (muszą się zgadzać!) ──────────────────────

DATA_PATH  = Path("data/mapped.csv")
SAVE_DIR   = Path("models/bert_category")
CM_PATH    = SAVE_DIR / "confusion_matrix.png"
MAX_LEN    = 256
BATCH_SIZE = 16
TEST_SPLIT = 0.1
SEED       = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Używane urządzenie: {DEVICE}")

# ── 1. Wczytanie zapisanego modelu i tokenizera ───────────────────────────────

tokenizer = BertTokenizerFast.from_pretrained(SAVE_DIR)
model = BertForSequenceClassification.from_pretrained(SAVE_DIR)
model.to(DEVICE)
model.eval()

# Mapowanie etykiet bierzemy z modelu, żeby na pewno było spójne z treningiem
label2id = model.config.label2id
id2label = {int(idx): lab for lab, idx in label2id.items()}
class_names = [id2label[i] for i in range(len(id2label))]
print(f"Klasy: {label2id}")

# ── 2. Wczytanie danych (tak samo jak w bert.py) ──────────────────────────────

def load_data(path: Path):
    texts, labels = [], []
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row["subject"] + " [SEP] " + row["body"]
            texts.append(text)
            labels.append(row["category"])
    return texts, labels

texts, labels = load_data(DATA_PATH)
label_ids = [label2id[l] for l in labels]

# ── 3. Odtworzenie tego samego zbioru testowego ───────────────────────────────
# Pierwszy split w bert.py oddziela zbiór testowy — wystarczy go powtórzyć.

_, texts_test, _, labels_test = train_test_split(
    texts, label_ids,
    test_size=TEST_SPLIT,
    random_state=SEED,
    stratify=label_ids,
)
print(f"Zbiór testowy: {len(texts_test)} próbek")

# ── 4. Tokenizacja i Dataset ──────────────────────────────────────────────────

enc_test = tokenizer(
    texts_test,
    max_length=MAX_LEN,
    truncation=True,
    padding="max_length",
    return_tensors="pt",
)

class TicketDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

test_loader = DataLoader(TicketDataset(enc_test, labels_test), batch_size=BATCH_SIZE)

# ── 5. Predykcja ──────────────────────────────────────────────────────────────

all_preds, all_labels = [], []
with torch.no_grad():
    for batch in test_loader:
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        token_type_ids = batch["token_type_ids"].to(DEVICE)

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        ).logits

        all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
        all_labels.extend(batch["labels"].numpy())

print("\n=== Raport klasyfikacji (zbiór testowy) ===")
print(classification_report(all_labels, all_preds, target_names=class_names))

# ── 6. Macierz pomyłek ────────────────────────────────────────────────────────

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
)
plt.title("Macierz Pomyłek: BERT")
plt.ylabel("Rzeczywista kategoria")
plt.xlabel("Przewidziana kategoria")

plt.savefig(CM_PATH, bbox_inches="tight")
print(f"\nZapisano macierz pomyłek w: {CM_PATH}")
