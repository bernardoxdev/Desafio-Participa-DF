import os
import re
import pandas as pd
import spacy
import joblib

from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from brutils import is_valid_cep, is_valid_cnpj, is_valid_cpf, is_valid_email, is_valid_license_plate, is_valid_phone, is_valid_pis, is_valid_voter_id

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "models", "context_model.joblib")
MODEL_CLAS_PATH = os.path.join(BASE_DIR, "models", "class_model.joblib")

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

def load_manual_train_dataset() -> pd.DataFrame:
    texts = [
        "Solicito pagamento de débito financeiro em aberto.",
        "Pedido de restituição de imposto pago indevidamente.",
        "Solicito revisão de cobrança referente ao imóvel.",
        "Histórico de consumo financeiro do imóvel solicitado.",
        "Esclarecimentos sobre multa financeira aplicada.",
        "Estou em tratamento médico contínuo e preciso de isenção.",
        "Encaminho laudo médico para análise de benefício.",
        "Paciente solicita afastamento por motivo de saúde.",
        "Solicito avaliação médica para concessão de direito.",
        "Possuo condição médica crônica e necessito licença.",
        "Solicito vaga em creche para criança menor de idade.",
        "Pedido de matrícula escolar para aluno menor.",
        "Criança necessita atendimento prioritário.",
        "Gostaria de informações sobre vacinação infantil.",
        "Solicito transferência escolar para menor.",
        "Requeiro cópia integral de processo administrativo.",
        "Solicito acesso a processo registrado no sistema.",
        "Gostaria de saber o andamento de processo legal.",
        "Pedido formal de abertura de processo administrativo.",
        "Solicito informações sobre denúncia registrada.",
        "Encaminho documentos pessoais para cadastro.",
        "Solicito atualização cadastral com dados pessoais.",
        "Meu nome completo consta incorreto no cadastro.",
        "Encaminho CPF e RG para validação de identidade.",
        "Solicito correção de dados cadastrais pessoais."
    ]

    labels = [
        "FINANCIAL", "FINANCIAL", "FINANCIAL", "FINANCIAL", "FINANCIAL",
        "HEALTH", "HEALTH", "HEALTH", "HEALTH", "HEALTH",
        "CHILD", "CHILD", "CHILD", "CHILD", "CHILD",
        "LEGAL", "LEGAL", "LEGAL", "LEGAL", "LEGAL",
        "IDENTITY", "IDENTITY", "IDENTITY", "IDENTITY", "IDENTITY"
    ]

    return pd.DataFrame({
        "text": texts,
        "label": labels
    })    

def train_context_model(csv_path, save = True):
    df = load_multiple_datasets(csv_path)
    train_df = load_manual_train_dataset()

    X_train = df["text"]
    y_train = df["label"]
    
    X_test = train_df["text"]
    y_test = train_df["label"]
    
    print(y_train.value_counts())
    print(y_test.value_counts())

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1,1),
            max_features=3000,  
            min_df=3,
            max_df=0.85,
            sublinear_tf=True
        )),
        ("clf", LogisticRegression(
            C=0.3,
            penalty="l2",
            class_weight="balanced",
            max_iter=1000
        ))
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

def load_multiple_datasets(csv_paths: list) -> pd.DataFrame:
    dfs = []

    for path in csv_paths:
        df = pd.read_csv(path)
        dfs.append(df)

    df_final = pd.concat(dfs, ignore_index=True)
    return df_final

def load_manual_train_dataset_clas() -> pd.DataFrame:
    setencs = [
        "Não consigo acessar minha conta por causa dos dados incorretos.",
        "Meu CPF é 123.456.789-09, preciso atualizar o cadastro.",
        "O CNPJ da empresa é 12.345.678/0001-95.",
        "Pode enviar o boleto para o e-mail joao.silva@gmail.com?",
        "O número do processo é 0701234-56.2023.8.07.0001.",
        "Meu telefone para contato é (61) 99876-5432.",
        "O CEP da entrega é 70297-400.",
        "A placa do carro é ABC1D23.",
        "Meu PIS é 123.45678.90-1.",
        "Título de eleitor: 1234 5678 9012.",
        "Preciso confirmar meus dados pessoais no sistema.",
        "Atualize minhas informações cadastrais, por favor.",
        "Houve um erro na validação do meu documento.",
        "O sistema pediu um número de identificação válido.",
        "É necessário informar os dados do responsável legal.",
        "O CPF é um documento utilizado no Brasil.",
        "Como funciona o cálculo do CNPJ?",
        "Explique o que é um CEP.",
        "Esse sistema valida documentos automaticamente.",
        "O modelo identifica dados sensíveis em textos.",
        "A API utiliza regex para validação de padrões.",
        "O erro 404 ocorreu durante a requisição.",
        "A versão do sistema é 1.12.2.",
        "O pedido número 12345 foi processado.",
        "O usuário marcou 15 pontos no ranking.",
        "O servidor está rodando na porta 8080.",
        "meu cpf eh 12345678909",
        "manda msg no zap 61998765432",
        "me chama no email teste123@gmail.com",
        "meu doc ta errado no sistema",
        "meu cadastro deu ruim",
        "O campo CPF não deve aceitar letras.",
        "O sistema bloqueia CNPJs inválidos.",
        "Não armazene dados pessoais em logs.",
        "A validação de telefone falhou no teste unitário.",
        "Preciso revisar minhas informações.",
        "O sistema solicitou um número.",
        "Meus dados estão incorretos.",
        "O formulário exige preenchimento obrigatório."
    ]

    y_true = [
        "PII","PII","PII","PII","PII","PII","PII","PII","PII","PII",
        "PII","PII","PII","PII","PII",
        "NON_PII","NON_PII","NON_PII","NON_PII","NON_PII","NON_PII",
        "NON_PII","NON_PII","NON_PII","NON_PII","NON_PII",
        "PII","PII","PII","PII","PII",
        "NON_PII","NON_PII","NON_PII","NON_PII","NON_PII",
        "NON_PII","NON_PII","NON_PII"
    ]

    return pd.DataFrame({
        "text": setencs,
        "label": y_true
    })

def train_class_model(csv_paths: list, save: bool = True):
    df = load_multiple_datasets(csv_paths)
    train_df = load_manual_train_dataset_clas()
    
    X_train = df["text"]
    y_train = df["label"]
    
    X_test = train_df["text"]
    y_test = train_df["label"]
    
    print(y_train.value_counts())
    print(y_test.value_counts())

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1,1),
            max_features=3000,  
            min_df=3,
            max_df=0.85,
            sublinear_tf=True
        )),
        ("clf", LogisticRegression(
            C=0.3,
            penalty="l2",
            class_weight="balanced",
            max_iter=1000
        ))
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
        joblib.dump(model, MODEL_CLAS_PATH)
        print(f"💾 Modelo salvo em: {MODEL_CLAS_PATH}")
    
    return model

def load_class_model():
    if os.path.exists(MODEL_CLAS_PATH):
        print("📦 Modelo carregado do disco")
        return joblib.load(MODEL_CLAS_PATH)
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

def detect_pii(text, context_model, clas_model):
    result = {
        "regex": [],
        "brutils": [],
        "entities": [],
        "model": [],
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
    
    result["model"] = str(clas_model.predict([text])[0])

    if context in ["HEALTH", "CHILD"]:
        result["risk"] = "HIGH"
    elif len(result["model"]) + len(result["regex"]) + len(result["brutils"]) + len(result["entities"]) >= 2:
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

def process_text_row(text: str, context_model: object, clas_model: object) -> dict:
    pii = detect_pii(text, context_model, clas_model)
    analysis = analyze_text_multilabel(text, context_model)
    masked = mask_text(text)

    return {
        "texto_masked": masked,
        "pii_regex": ",".join(pii["regex"]),
        "pii_brutils": ",".join(pii["brutils"]),
        "pii_entities": ",".join(
            [f"{e['text']}:{e['label']}" for e in pii["entities"]]
        ),
        "pii_clas": pii["model"],
        "context": str(pii["context"]),
        "risk_pii": pii["risk"],
        "lgpd_score_global": analysis["global"]["score"],
        "lgpd_risk_global": analysis["global"]["risk"]
    }

if __name__ == "__main__":
    pass