import os
import re
import pandas as pd
import spacy
import joblib

from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from brutils import is_valid_cep, is_valid_cnpj, is_valid_cpf, is_valid_email, is_valid_license_plate, is_valid_phone, is_valid_pis, is_valid_voter_id

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "models", "context_model.joblib")

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

REGEX_PII = {
    "PROCESS": r"\b\d{4,}-\d{2,}/\d{4}-\d{2}\b",
    "CPF": r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b",
    "CPF_SIMPLES": r"\b\d{11}\b",
    "EMAIL": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
    "TELEFONE": r"\b(\(?\d{2}\)?\s?)?\d{4,5}-?\d{4}\b",
    "RG": r"\b\d{1,2}\.\d{3}\.\d{3}-?\d{1}\b",
    "ENDERECO": r"\b(rua|avenida|av\.|travessa|alameda)\b",
    "SEI_PROCESS": r"\b\d{5}\s*-\s*\d{8}\s*/\s*\d{4}\s*-\s*\d{2}\b"
}

PII_SCORES = {
    "CPF": 10,
    "CNPJ": 8,
    "VOTER_ID": 10,
    "PIS": 9,
    "EMAIL": 6,
    "PHONE": 6,
    "CEP": 4,
    "LICENSE_PLATE": 6,
    "LEGAL_PROCESS": 7,
    "NAME": 6,
    "HEALTH": 10,
    "CHILD": 10,
    "LEGAL": 7,
    "FINANCIAL": 7,
    'SEI_PROCESS': 7
}

nlp = spacy.load("pt_core_news_lg")

def detect_ner(text):
    doc = nlp(text)
    return [
        {"text": ent.text, "label": ent.label_}
        for ent in doc.ents
        if ent.label_ in ["PER", "LOC"]
    ]

def train_context_model(csv_path, save = True):
    df = pd.read_csv(csv_path)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )
    
    print(y_train.value_counts())
    print(y_test.value_counts())

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1,2),
            max_features=15000
        )),
        ("clf", MultinomialNB())
    ])

    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, df["text"], df["label"], cv=skf)

    print("Accuracy média:", scores.mean())

    print(f"✔ Accuracy do classificador de contexto: {acc:.2f}")
    
    if save:
        joblib.dump(model, MODEL_PATH)
        print(f"💾 Modelo salvo em: {MODEL_PATH}")
    
    return model

def load_context_model():
    if os.path.exists(MODEL_PATH):
        print("📦 Modelo carregado do disco")
        return joblib.load(MODEL_PATH)
    else:
        raise FileNotFoundError("Modelo não encontrado. Treine o modelo primeiro.")

def detect_brutils(text: str) -> list:
    found = set()

    tokens = re.findall(r"\b[\w\./\-@]+\b", text)

    for t in tokens:
        if is_valid_cpf(t):
            found.add("CPF")
        elif is_valid_cnpj(t):
            found.add("CNPJ")
        elif is_valid_cep(t):
            found.add("CEP")
        elif is_valid_phone(t):
            found.add("PHONE")
        elif is_valid_email(t):
            found.add("EMAIL")
        elif is_valid_license_plate(t):
            found.add("LICENSE_PLATE")
        elif is_valid_pis(t):
            found.add("PIS")
        elif is_valid_voter_id(t):
            found.add("VOTER_ID")

    return list(found)

def detect_pii(text, context_model):
    result = {
        "regex": [],
        "brutils": [],
        "entities": [],
        "context": None,
        "risk": "LOW"
    }

    for name, pattern in REGEX_PII.items():
        if re.search(pattern, text):
            result["regex"].append(name)

    result["brutils"] = detect_brutils(text)

    entities = detect_ner(text)
    if entities:
        result["entities"] = entities

    context = str(context_model.predict([text])[0])
    result["context"] = context

    if context in ["HEALTH", "CHILD"]:
        result["risk"] = "HIGH"
    elif len(result["regex"]) + len(result["brutils"]) + len(result["entities"]) >= 2:
        result["risk"] = "MEDIUM"

    return result

def mask_text(text):
    masked = text

    patterns = REGEX_PII

    for name, pattern in patterns.items():
        masked = re.sub(pattern, f"[{name}]", masked)

    doc = nlp(masked)
    for ent in doc.ents:
        if ent.label_ == "PER":
            masked = masked.replace(ent.text, "[NOME]")

    return masked

def calculate_lgpd_score(pii_types: list) -> dict:
    total = 0
    details = {}

    for pii in pii_types:
        score = PII_SCORES.get(pii, 3)
        details[pii] = score
        total += score

    if total >= 20:
        level = "HIGH"
    elif total >= 10:
        level = "MEDIUM"
    else:
        level = "LOW"

    return {
        "score": total,
        "risk_level": level,
        "details": details
    }
    
def split_sentences(text):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]

def classify_sentence(sentence, context_model):
    labels = set()

    br_labels = detect_brutils(sentence)
    labels.update(br_labels)

    ents = detect_ner(sentence)
    if ents:
        labels.add("NAME")

    context = str(context_model.predict([sentence])[0])
    labels.add(context)

    return list(labels)

def analyze_text_multilabel(text, context_model):
    sentences = split_sentences(text)

    analysis = []
    global_labels = set()

    for sent in sentences:
        labels = classify_sentence(sent, context_model)
        global_labels.update(labels)

        score_info = calculate_lgpd_score(labels)

        analysis.append({
            "sentence": sent,
            "labels": labels,
            "lgpd_score": score_info["score"],
            "risk": score_info["risk_level"]
        })

    global_score = calculate_lgpd_score(global_labels)

    return {
        "sentences": analysis,
        "global": {
            "labels": list(global_labels),
            "score": global_score["score"],
            "risk": global_score["risk_level"]
        }
    }

def process_text_row(text: str, context_model: object) -> dict:
    pii = detect_pii(text, context_model)
    analysis = analyze_text_multilabel(text, context_model)
    masked = mask_text(text)

    return {
        "texto_masked": masked,
        "pii_regex": ",".join(pii["regex"]),
        "pii_brutils": ",".join(pii["brutils"]),
        "pii_entities": ",".join(
            [f"{e['text']}:{e['label']}" for e in pii["entities"]]
        ),
        "context": str(pii["context"]),
        "risk_pii": pii["risk"],
        "lgpd_score_global": analysis["global"]["score"],
        "lgpd_risk_global": analysis["global"]["risk"]
    }

if __name__ == "__main__":
    pass