"""
bert.py

Trening modelu BERT do klasyfikacji zgłoszeń na kategorie:
  - 'technical' (wsparcie techniczne)
  - 'support'   (obsługa klienta, billing, zwroty itp.)

Dane wejściowe: data/mapped.csv (wynik mapdata.py)
Wejście modelu: pole 'subject' + 'body' (tekst zgłoszenia)
Wyjście modelu: pole 'category' (etykieta klasy)

Używamy bert-base-multilingual-cased ponieważ dane są w języku DE i EN.
"""

import csv
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pathlib import Path

# ── Hiperparametry ────────────────────────────────────────────────────────────

MODEL_NAME   = "bert-base-multilingual-cased"
DATA_PATH    = Path("data/mapped.csv")
SAVE_DIR     = Path("models/bert_category")
MAX_LEN      = 256
BATCH_SIZE   = 16
EPOCHS       = 3
LR           = 2e-5
WARMUP_RATIO = 0.1
VAL_SPLIT    = 0.1
TEST_SPLIT   = 0.1
SEED         = 42

# ── Urządzenie (GPU lub CPU) ──────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Używane urządzenie: {DEVICE}")

# ── 1. Wczytanie danych ───────────────────────────────────────────────────────

def load_data(path: Path) -> tuple[list[str], list[str]]:
    """Wczytuje CSV i zwraca (teksty, etykiety)."""
    texts, labels = [], []
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row["subject"] + " [SEP] " + row["body"]
            texts.append(text)
            labels.append(row["category"])
    return texts, labels

texts, labels = load_data(DATA_PATH)
print(f"Wczytano {len(texts)} próbek.")

# ── 2. Kodowanie etykiet ──────────────────────────────────────────────────────

label2id = {label: idx for idx, label in enumerate(sorted(set(labels)))}
id2label = {idx: label for label, idx in label2id.items()}
label_ids = [label2id[l] for l in labels]
NUM_LABELS = len(label2id)
print(f"Klasy: {label2id}")

# ── 3. Podział na train / val / test ─────────────────────────────────────────

texts_tv, texts_test, labels_tv, labels_test = train_test_split(
    texts, label_ids,
    test_size=TEST_SPLIT,
    random_state=SEED,
    stratify=label_ids,
)

val_ratio = VAL_SPLIT / (1 - TEST_SPLIT)
texts_train, texts_val, labels_train, labels_val = train_test_split(
    texts_tv, labels_tv,
    test_size=val_ratio,
    random_state=SEED,
    stratify=labels_tv,
)

print(f"Train: {len(texts_train)}, Val: {len(texts_val)}, Test: {len(texts_test)}")

# ── 4. Tokenizer ──────────────────────────────────────────────────────────────

tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

def tokenize(texts: list[str]) -> dict:
    """Tokenizuje listę tekstów i zwraca tensory gotowe dla BERT."""
    return tokenizer(
        texts,
        max_length=MAX_LEN,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

enc_train = tokenize(texts_train)
enc_val   = tokenize(texts_val)
enc_test  = tokenize(texts_test)

# ── 5. Dataset ────────────────────────────────────────────────────────────────

class TicketDataset(Dataset):
    """Opakowanie danych w klasę Dataset wymaganą przez DataLoader PyTorch."""

    def __init__(self, encodings: dict, labels: list[int]):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

train_dataset = TicketDataset(enc_train, labels_train)
val_dataset   = TicketDataset(enc_val,   labels_val)
test_dataset  = TicketDataset(enc_test,  labels_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

# ── 6. Model ──────────────────────────────────────────────────────────────────

model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    id2label=id2label,
    label2id=label2id,
)
model.to(DEVICE)

# ── 7. Optymalizator i scheduler ─────────────────────────────────────────────

optimizer = AdamW(model.parameters(), lr=LR)

total_steps = EPOCHS * len(train_loader)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(WARMUP_RATIO * total_steps),
    num_training_steps=total_steps,
)

# ── 8. Pętla treningowa ───────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, scheduler) -> float:
    """Jedna epoka treningu. Zwraca średnią stratę."""
    model.train()
    total_loss = 0.0

    for batch in loader:
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        token_type_ids = batch["token_type_ids"].to(DEVICE)
        label_batch    = batch["labels"].to(DEVICE)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=label_batch,
        )

        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

    return total_loss / len(loader)


def evaluate(model, loader) -> tuple[float, str]:
    """Ewaluacja na zbiorze val lub test. Zwraca (strata, raport klasyfikacji)."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            token_type_ids = batch["token_type_ids"].to(DEVICE)
            label_batch    = batch["labels"].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=label_batch,
            )

            total_loss += outputs.loss.item()
            preds = outputs.logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label_batch.cpu().numpy())

    avg_loss = total_loss / len(loader)
    report = classification_report(
        all_labels, all_preds,
        target_names=list(label2id.keys()),
    )
    return avg_loss, report


# ── 9. Trening ────────────────────────────────────────────────────────────────

print("\n=== Start treningu ===")
for epoch in range(1, EPOCHS + 1):
    train_loss = train_epoch(model, train_loader, optimizer, scheduler)
    val_loss, val_report = evaluate(model, val_loader)
    print(f"\nEpoka {epoch}/{EPOCHS}")
    print(f"  Train loss: {train_loss:.4f}  |  Val loss: {val_loss:.4f}")
    print("  Walidacja:\n", val_report)

# ── 10. Test końcowy ──────────────────────────────────────────────────────────

print("\n=== Wyniki na zbiorze testowym ===")
test_loss, test_report = evaluate(model, test_loader)
print(f"Test loss: {test_loss:.4f}")
print(test_report)

# ── 11. Zapis modelu ──────────────────────────────────────────────────────────

SAVE_DIR.mkdir(parents=True, exist_ok=True)
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print(f"\nModel zapisany w: {SAVE_DIR}")

# ── 12. Przykład inferencji ───────────────────────────────────────────────────

def predict(text: str) -> str:
    """Przewiduje kategorię dla nowego tekstu zgłoszenia."""
    model.eval()
    enc = tokenizer(
        text,
        max_length=MAX_LEN,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
    class_id = logits.argmax(dim=-1).item()
    return id2label[class_id]

example = "Server is down [SEP] Our production server stopped responding after the latest update."
print(f"\nPrzykład inferencji:")
print(f"  Tekst:     {example}")
print(f"  Predykcja: {predict(example)}")
