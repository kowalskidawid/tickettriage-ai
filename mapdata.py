"""
mapdata.py

Wczytuje data/raw.csv, klasyfikuje każde zgłoszenie jako 'technical' lub 'support',
balansuje liczbę wierszy tak, aby każda grupa (język × kategoria) miała ten sam
rozmiar (undersampling do najmniejszej grupy) i zapisuje wynik do
data/mapped.csv.

Logika klasyfikacji (na podstawie kolumny 'queue'):
  technical → Technical Support, IT Support, Product Support,
               Service Outages and Maintenance
  support   → Returns and Exchanges, Billing and Payments, Sales and Pre-Sales,
               Customer Service, Human Resources, General Inquiry
"""

import csv
import random
from collections import defaultdict
from pathlib import Path

RANDOM_SEED = 42

TECHNICAL_QUEUES = {
    "Technical Support",
    "IT Support",
    "Product Support",
    "Service Outages and Maintenance",
}

INPUT_PATH = Path("data/raw.csv")
OUTPUT_PATH = Path("data/mapped.csv")

OUTPUT_COLUMNS = [
    "subject",
    "body",
    "answer",
    "type",
    "queue",
    "priority",
    "language",
    "category",
]


def classify(queue: str) -> str:
    return "technical" if queue in TECHNICAL_QUEUES else "support"


def load_and_classify(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["category"] = classify(row["queue"])
            rows.append(row)
    return rows


def balance(rows: list[dict], seed: int = RANDOM_SEED) -> list[dict]:
    """Zmniejsza każdą grupę (język, kategoria) do rozmiaru najmniejszej z nich."""
    by_group: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        by_group[(row["language"], row["category"])].append(row)

    min_count = min(len(g) for g in by_group.values())
    print("Row counts per (language, category) before balancing:")
    for key, group in sorted(by_group.items()):
        print(f"  {key}: {len(group)}")
    print(f"Balancing each group to {min_count} rows.")

    rng = random.Random(seed)
    balanced = []
    for group in by_group.values():
        balanced.extend(rng.sample(group, min_count))

    return balanced


def save(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    print(f"Reading {INPUT_PATH} ...")
    rows = load_and_classify(INPUT_PATH)
    print(f"Loaded {len(rows)} rows.")

    tech_count = sum(1 for r in rows if r["category"] == "technical")
    support_count = len(rows) - tech_count
    print(f"  technical: {tech_count}, support: {support_count}")

    balanced = balance(rows)

    tech_count = sum(1 for r in balanced if r["category"] == "technical")
    support_count = len(balanced) - tech_count
    print(f"After balancing: {len(balanced)} rows total")
    print(f"  technical: {tech_count}, support: {support_count}")

    save(balanced, OUTPUT_PATH)
    print(f"Saved to {OUTPUT_PATH}.")


if __name__ == "__main__":
    main()
