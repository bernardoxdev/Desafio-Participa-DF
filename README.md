# 1º Hackathon em Controle Social: Desafio Participa DF

## Autores

- https://github.com/bernardoxdev [IA]
- https://github.com/Diegovsky [Web]
- https://github.com/Qaddish [Web]

## O desafio: Desenvolver um modelo capaz de identificar automaticamente pedidos públicos que contenham dados pessoais.

Foi disponibilizado um modelo dos dados que está disponível dentro de `backend/data/amostra_data.xlsx`

# Nossa solução:

    Para resolver o problema primeiro precisamos entender o que é um "dado pessoal" de acordo com a LGPD, dados pessoais são dados como:

- Nome de pessoas
- Email
- Telefone
- CPF / RG
- CNH
- Título de Eleitor
- OAB
- Número de processo SEI
- Número de ocorrência policial
- Protocolo LAI
- Número de contrato
- Número de inscrição de consumo
- Número de matrícula funcional
- Dados jurídicos e criminais

    Iremos utilizar como camadas da solução híbrida o seguinte:

```
Texto
 ├── Regex (CPF, email, telefone, processo)
 ├── Brutils (Dados brasileiros geral	)
 ├── NER (nomes, locais, organizações)
 ├── Classificador de contexto (saúde, jurídico, criança)
 └── Agregador de risco (LGPD score)
```

    Inicialmente o sistema cria um modelo para classificar o texto com base em técnicas de`TfidfVectorizer` e `Logistic Regression`, ele está sendo treinado a partir de datasets gerados por LLMs (Grandes Modelos de Linguagem) e dados tratados por pipelines de REGEX e NER.

Como funciona a pipeline de NLP:

```
Texto Cru -> TF - IDF (Vetorização) -> Regessão Logística (Classificação)
```

Métricas usadas:

```
TfidfVectorizer(
	ngram_range=(1, 1),
	max_feature=3000,
	min_df=3,
	max_df=0.85,
	sublinear_tf=True
)
```

    No TF-IDF estamos utilizando apenas unigramas (uma palavra), limitamos a um range de 3000 palavras, ignoramos palavras que aparecem em menos de 3 documentos, removemos  stopwords e evitamos palavras repetidas dominem o modelo com o`TF = 1 + log(TF).`

```
LogisticRegression(
	C=0.3,
	class_weight="balaced",
	max_iter=1000
)
```

    Na regressão logística fazemos com que se tenha uma forte regularização (0.3) como nosso texto possui muito ruído, deixamos o próprio programa ajustar o peso das classes (PII << NON_PIII [ou vice-versa)) e garantimos a convergência.

```
StratifiedKFold(
	n_splits=5,
	shuffle=True,
	random_state=42
)
```

    Divide o dataset em 5 folds, treina e teste 5 vezes, com cada fold mantendo a proporação de classes.

    Alguns conceitos importantes voltados a Regressão Logística são a "regularização" e o "max_iter garantir convergência". Primeiro precisamos entender o que a regressão logística em L2 tenta minimizar:`Erro + J + Complexidade`

- Erro -> Criar previsões
- Complexidade -> Pesos muito grandes

L2: \\[L_2 = \sum_{i} w_i^2\\]

    Resumindo: Penalidade grande a pesos grandes e modelo mais suave a pesos pequenos

    A regularização matematicamente falando é`C = 1 / J`. Agora qual a diferença entre L1 e L2 -> L1 zera os pesos e L2 diminui os pesos (seleção x generalização). O max_iter gera tempo para a regularização agir e gradiente estabilizar.

    Após os contextos serem gerados é passado para análise se um documento é PII ou NON_PII, as métricas e racícionio são os mesmos do primeiro modelo. Com isso, é passado para análise de o que está contido e o mascaramento do texto. Antes vamos passar pelos números dos modelos:

- Conext: (TR) 200 -> (TS) 5

|              | Precision | Recall | F1-Score | Support |
| ------------ | --------- | ------ | -------- | ------- |
| CHILD        | 1.0       | 1.0    | 1.0      | 5       |
| FINANCIAL    | 0.83      | 1.00   | 0.93     | 5       |
| HEALTH       | 1.00      | 1.00   | 1.00     | 5       |
| IDENTITY     | 1.00      | 0.80   | 0.89     | 5       |
| LEGAL        | 1.00      | 1.00   | 1.00     | 5       |
| ---          | ---       | ---    | ---      | ---     |
| Accuracy     | ---       | ---    | 0.96     | 25      |
| Macro AVG    | 0.97      | 0.96   | 0.96     | 25      |
| Weigthed AVG | 0.97      | 0.96   | 0.96     | 25      |

Accuracy Média: 0.96

- Classification: (TR) 625 -> (TS) 20

|              | Precision | Recall | F1-Score | Support |
| ------------ | --------- | ------ | -------- | ------- |
| NON_PII      | 0.58      | 0.58   | 0.58     | 19      |
| PII          | 0.60      | 0.60   | 0.60     | 20      |
| ---          | ---       | ---    | ---      | ---     |
| Accuracy     | ---       | ---    | 0.59     | 39      |
| Macro AVG    | 0.59      | 0.59   | 0.59     | 39      |
| Weigthed AVG | 0.59      | 0.59   | 0.59     | 39      |

Accuracy Média: 0.59

    Como existem outros 3 classificadores antes do modelo, sua accuracy de 0.59 não vai ser tão prejudicial ao projeto. Agora vamos falar dos nossos outros 3 classificadores e do mascaramento do texto.

    As análises que temos são: REGEX, Brutils, Spacy. Com cada uma completando a outra em seus pontos fracos:

- Regex:
  - Pontos fortes: Extremamente rápido; Determinístico; Não depende de modelo ou treinamento; Fácil de auditar;
  - Pontos fracos: Alta taxa de falsos positivos; Não entende contexto; Não válida semanticamente;
- Brutils:
  - Pontos fortes: Especializado em documentos reais do Brasil; Validação por dígito verificador; Reduz falsos positivos;
  - Pontos fracos: Recall baixo; Não detecta contexto; Falha em textos informais ou com erros;
- Spacy:
  - Pontos fortes: Detecta nome de pessoa sem padrão fixo; Funciona mesmo sem documentação expliícita;
  - Pontos fracos: Falsos positivos frequentes; Custo computacional elevado; Modelo não treinado especificamente para LGPD
- Resumo:

| Técnica | Melhor para                | Falha em               |
| -------- | -------------------------- | ---------------------- |
| Regex    | Padrões fixos             | Contexto e Significado |
| Brutils  | Documentos reais           | Texto informal         |
| Spacy    | Nomes e PII implícita     | Ambiguidade            |
| Modelo   | Intenção  e Significado | Determinismo           |

    Já quando vamos falar sobre o mascaramento do texto estamos falando do nosso 1° modelo + REGEX + Spacy, os 3 fazem uma junção onde identificam as partes do texto que são dados válidos retornando mascarado. Por exemplo:

`"meu nome é walter rodrigues cruz, no dia 25..." -> "meu nome é [NOME], no dia 25..."`
