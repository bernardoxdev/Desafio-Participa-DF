# 1º Hackathon em Controle Social: Desafio Participa DF

## Autores

- https://github.com/bernardoxdev
- https://github.com/Diegovsky
- https://github.com/Qaddish

## O desafio: Desenvolver um modelo capaz de identificar automaticamente pedidos públicos que contenham dados pessoais.

Foi disponibilizado um modelo dos dados que está disponível dentro de `data/data.xlsx`

# Nossa solução:

Para resolver o problema primeiro precisamos entender o que é um "dado pessoal" de acordo com a LGPD, dados pessoais são dados como:

- Nome de pessoas
- Email
- Telefone
- CPF / RG
- CNH
- Título de Eleitor
- Matrícula funcional
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

Com essas várias camadas na solução diminui o risco que existam erros de classificação das informações, assim fazendo com que todos os dados sensíveis em um texto sejam identificados e relatados.
