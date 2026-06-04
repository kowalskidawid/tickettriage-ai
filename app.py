"""
app.py

Prosta aplikacja webowa (Flask) do klasyfikacji zgłoszeń.
Użytkownik wpisuje temat i treść zgłoszenia, a strona pokazuje predykcję
każdego dostępnego modelu (Naive Bayes, regresja liniowa, BERT).

Uruchomienie:
    pip install -r requirements.txt
    python app.py
    -> http://127.0.0.1:5000
"""

import sys
from pathlib import Path

from flask import Flask, render_template, request

NB_PATH      = Path("models/naive_bayes.joblib")
LINREG_PATH  = Path("models/linear_regression.joblib")
BERT_DIR     = Path("models/bert_category")
LSTM_PATH    = Path("models/lstm/Test1.keras")
LSTM_TOKENIZER_PATH = Path("models/lstm/tokenizer_Test1.pickle")

LINREG_THRESHOLD = 0.5
LSTM_MAX_LEN     = 200  # zgodne z max_len z LSTM.py

app = Flask(__name__)

# ── Ładowanie modeli (raz, przy starcie) ──────────────────────────────────────

def load_naive_bayes():
    if not NB_PATH.exists():
        return None
    import joblib
    return joblib.load(NB_PATH)


def load_linear_regression():
    if not LINREG_PATH.exists():
        return None
    import joblib
    return joblib.load(LINREG_PATH)


def load_bert():
    if not BERT_DIR.exists():
        return None
    import torch
    from transformers import BertTokenizerFast, BertForSequenceClassification
    tokenizer = BertTokenizerFast.from_pretrained(BERT_DIR)
    model = BertForSequenceClassification.from_pretrained(BERT_DIR)
    model.eval()
    return {"tokenizer": tokenizer, "model": model, "torch": torch}


def load_lstm():
    if not LSTM_PATH.exists() or not LSTM_TOKENIZER_PATH.exists():
        return None
    import os
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    import pickle
    try:
        from tensorflow.keras.models import load_model
        from tensorflow.keras.preprocessing.sequence import pad_sequences
    except ImportError:
        # brak TensorFlow - model LSTM po prostu zgloszony jako niedostepny
        print("LSTM skipped: TensorFlow not installed (pip install tensorflow).")
        return None
    with LSTM_TOKENIZER_PATH.open("rb") as f:
        tokenizer = pickle.load(f)
    model = load_model(LSTM_PATH)
    return {"model": model, "tokenizer": tokenizer, "pad_sequences": pad_sequences}


nb_model     = load_naive_bayes()
linreg_model = load_linear_regression()
bert_model   = load_bert()
lstm_model   = load_lstm()

# ── Predykcje per model ───────────────────────────────────────────────────────

def predict_naive_bayes(subject: str, body: str) -> dict:
    if nb_model is None:
        return {"available": False}
    text = subject + " " + body
    pred = nb_model.predict([text])[0]
    proba = nb_model.predict_proba([text])[0]
    proba_dict = {cls: round(float(p), 4) for cls, p in zip(nb_model.classes_, proba)}
    return {"available": True, "prediction": pred, "scores": proba_dict}


def predict_linear_regression(subject: str, body: str) -> dict:
    if linreg_model is None:
        return {"available": False}
    text = subject + " " + body
    score = float(linreg_model.predict([text])[0])
    # model zapisany w linear_regression.py koduje support=0, technical=1
    classes = sorted({"support", "technical"})
    pred = classes[int(score >= LINREG_THRESHOLD)]
    return {
        "available": True,
        "prediction": pred,
        "scores": {"wynik regresji": round(score, 4), "próg": LINREG_THRESHOLD},
    }


def predict_bert(subject: str, body: str) -> dict:
    if bert_model is None:
        return {"available": False}
    torch = bert_model["torch"]
    tokenizer = bert_model["tokenizer"]
    model = bert_model["model"]
    text = subject + " [SEP] " + body
    enc = tokenizer(
        text,
        max_length=256,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    with torch.no_grad():
        logits = model(**enc).logits
    probs = torch.softmax(logits, dim=-1)[0]
    id2label = model.config.id2label
    scores = {id2label[i]: round(float(probs[i]), 4) for i in range(len(probs))}
    pred = id2label[int(logits.argmax(dim=-1).item())]
    return {"available": True, "prediction": pred, "scores": scores}


def predict_lstm(subject: str, body: str) -> dict:
    if lstm_model is None:
        return {"available": False}
    tokenizer = lstm_model["tokenizer"]
    model = lstm_model["model"]
    pad_sequences = lstm_model["pad_sequences"]
    # ten sam format tekstu co przy treningu (LSTM.py); brak pola 'answer' przy inferencji
    text = "TEMAT: " + subject + " TRESC: " + body + " ODPOWIEDZ: "
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=LSTM_MAX_LEN)
    p_tech = float(model.predict(padded, verbose=0)[0][0])  # sigmoid -> P(technical)
    classes = sorted({"support", "technical"})  # support=0, technical=1
    pred = classes[int(p_tech >= 0.5)]
    return {
        "available": True,
        "prediction": pred,
        "scores": {"support": round(1.0 - p_tech, 4), "technical": round(p_tech, 4)},
    }


MODELS = [
    ("Naive Bayes", predict_naive_bayes),
    ("Regresja liniowa", predict_linear_regression),
    ("BERT", predict_bert),
    ("LSTM", predict_lstm),
]

# ── Trasy ─────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET", "POST"])
def index():
    subject = ""
    body = ""
    results = None

    if request.method == "POST":
        subject = request.form.get("subject", "").strip()
        body = request.form.get("body", "").strip()
        if subject or body:
            results = []
            for name, fn in MODELS:
                out = fn(subject, body)
                out["name"] = name
                results.append(out)

    return render_template("index.html", subject=subject, body=body, results=results)


if __name__ == "__main__":
    app.run(debug=True)
