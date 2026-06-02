"""
lstm.py

Trening i eksperymenty z modelem LSTM do klasyfikacji zgłoszeń na kategorie:
  - 'technical' (Technical Support, IT Support, Product Support, Service Outages and Maintance)
  - 'support'   (Returns and Exchanges, Billing and Payments, Sales and Pre-Sales,Customer Service, Human Resources, General Inquiry)

Dane wejściowe: data/mapped.csv (wynik mapdata.py)
Wejście modelu: pole 'subject' + 'body' + 'answer' (połączony tekst zgłoszenia)
Wyjście modelu: pole 'category' zmapowane na wartości binarne (0 lub 1)
"""

import os
import warnings
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ── Wyciszenie ostrzeżeń ──────────────────────────────────────────────────────
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, SpatialDropout1D, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ── Hiperparametry (Baza) ─────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    'max_words': 10000,
    'max_len': 200,
    'embed_dim': 100,
    'lstm_units': 64,
    'spatial_dropout': 0.3,
    'lstm_dropout': 0.3,
    'rec_dropout': 0.3,
    'epochs': 15,
    'batch_size': 32,
    'test_size': 0.2,
    'random_state': 42
}
# Tworzenie struktury katalogów zapisu, jeśli nie istnieje
Path("models/lstm/confusion_matrix").mkdir(parents=True, exist_ok=True)

# ── 1. Globalne przygotowanie danych ──────────────────────────────────────────

print("Wczytywanie i czyszczenie danych...")
df = pd.read_csv("data/mapped.csv")

# Zbiorcze wypełnianie braków w tekście
cols_to_fill = ['subject', 'body', 'answer']
df[cols_to_fill] = df[cols_to_fill].fillna('')

df['full_text'] = "TEMAT: " + df['subject'] + " TRESC: " + df['body'] + " ODPOWIEDZ: " + df['answer']

# Mapowanie na wartości binarne i czyszczenie odrzutów
df['tokenized'] = df['category'].astype(str).str.strip().map({'support': 0, 'technical': 1})
df.dropna(subset=['tokenized'], inplace=True)

X_data = df['full_text'].tolist()
Y_data = df['tokenized'].values

print(f"Liczba poprawnych zgłoszeń gotowych do uczenia maszynowego: {len(X_data)}")

# ── 2. Główna funkcja uczenia maszynowego ────────────────────────────────────────────

def uruchom(nazwa: str, custom_config: dict = None):
    """
    Uruchamia proces treningu (lub ładuje zapisany na dysku model) dla
    zadanej konfiguracji parametrów. Generuje i zapisuje macierz pomyłek.
    """
    config = DEFAULT_CONFIG.copy()
    if custom_config:
        config.update(custom_config)

    print("Uruchamianie treningu")

    save_path = Path(f"models/lstm/{nazwa}.keras")
    token_save_path = Path(f"models/lstm/tokenizer_{nazwa}.pickle")

    # Ładowanie z dysku lub inicjalizacja nowego treningu
    if save_path.exists() and token_save_path.exists():
        print(f"Znaleziono zapisany model {nazwa}.")
        with open(token_save_path, 'rb') as handle:
            tokenizer = pickle.load(handle)
        model = load_model(save_path)
        czy_trenowac = False
    else:
        print(f"Brak modelu {nazwa}. Rozpoczynam nowy trening...")
        tokenizer = Tokenizer(num_words=config['max_words'])
        tokenizer.fit_on_texts(X_data)
        czy_trenowac = True

    # Sekwencjonowanie i podział danych
    X_padded = pad_sequences(tokenizer.texts_to_sequences(X_data), maxlen=config['max_len'])
    X_train, X_test, y_train, y_test = train_test_split(
        X_padded, Y_data, test_size=config['test_size'], random_state=config['random_state']
    )

    # Trening modelu
    if czy_trenowac:
        model = Sequential([
            Embedding(input_dim=config['max_words'], output_dim=config['embed_dim']),
            SpatialDropout1D(config['spatial_dropout']),
            LSTM(config['lstm_units'], dropout=config['lstm_dropout'], recurrent_dropout=config['rec_dropout']),
            Dense(1, activation='sigmoid')
        ])

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(f"Trening (Epoki: {config['epochs']}, Rozmiar paczki: {config['batch_size']})")

        model.fit(
            X_train, y_train,
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            validation_data=(X_test, y_test),
        )

        # Zapis plików treningowych
        model.save(save_path)
        with open(token_save_path, "wb") as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Zapisano model jako {save_path.name}")

    # Ewaluacja i wyniki
    _, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n---> Ostateczna dokładność dla '{nazwa}': {accuracy:.4f}")

    # Próg 0.5 dla predykcji binarnej
    y_pred = (model.predict(X_test, verbose=0) > 0.5).astype(int)

    # Wizualizacja i zapis macierzy pomyłek
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        confusion_matrix(y_test, y_pred), 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=['Support (0)', 'Technical (1)'],
        yticklabels=['Support (0)', 'Technical (1)']
    )
    plt.title(f'Macierz Pomyłek: {nazwa}')
    plt.ylabel('Rzeczywista kategoria')
    plt.xlabel('Przewidziana kategoria')

    cm_save_path = Path(f"models/lstm/confusion_matrix/confusion_matrix_{nazwa}.png")
    plt.savefig(cm_save_path, bbox_inches='tight')
    print(f"Zapisano wykres macierzy pomyłek w: {cm_save_path}")
    plt.close()

# ── 3. Testowanie i weryfikacja ───────────────────────────────────────────────

uruchom("Test1")
# Skuteczność na 5 prób: 0.7114 ± 0.0056

# uruchom("Test2", custom_config={'embed_dim': 32}) 
# Skuteczność na 5 prób: 0.7089 ± 0.0064

# uruchom("Test3", custom_config={'lstm_units': 128}) 
# Skuteczność na 5 prób: 0.7078 ± 0.0056

# uruchom("Test4", custom_config={'spatial_dropout': 0.6, 'lstm_dropout': 0.6, 'rec_dropout': 0.6}) 
# Skuteczność na 5 prób: 0.6881 ± 0.0032

# uruchom("Test5", custom_config={'batch_size': 16}) 
# Skuteczność na 5 prób 0.7186 ± 0.0033