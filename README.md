# Transcrição e Análise de Sentimentos de Áudio

Este projeto é uma aplicação desenvolvida com **Streamlit** que permite aos usuários realizar transcrição de áudio, tratamento do texto transcrito e análise de sentimentos. O foco é facilitar o processamento de entrevistas, feedbacks ou interações baseadas em áudio, transformando as informações em insights acionáveis.

---

## 📋 Funcionalidades

- **Upload de Áudio**: Carregue arquivos no formato `.wav` ou `.mp3`.
- **Transcrição de Áudio**: Utilize o modelo Whisper para transcrever o conteúdo do áudio.
- **Tratamento de Texto**: Realize a remoção de stopwords e pontuações utilizando **SpaCy**.
- **Análise de Sentimentos**: Identifique a polaridade do texto (positivo, negativo ou neutro) com **VADER Sentiment Analysis**.
- **Classificação de Falas**: Separe as falas do funcionário e do cliente de maneira automática.
- **Visualização de Sentimentos**: Exiba a proporção de sentimentos em um gráfico de barras horizontal.
- **Download de Resultados**: Baixe o áudio tratado e a análise de sentimentos em formato `.csv`.

---

## 🛠️ Tecnologias Utilizadas

- **Linguagem**: Python
- **Bibliotecas**:
  - [Streamlit](https://streamlit.io/): Interface interativa.
  - [Whisper](https://github.com/openai/whisper): Transcrição de áudio.
  - [SpaCy](https://spacy.io/): Tratamento de texto.
  - [NLTK](https://www.nltk.org/): Análise de sentimentos com **VADER**.
  - [Librosa](https://librosa.org/): Manipulação e ajuste de áudio.
  - [SoundFile](https://pysoundfile.readthedocs.io/): Processamento de áudio.
  - [Matplotlib](https://matplotlib.org/) e [Seaborn](https://seaborn.pydata.org/): Visualização de dados.
  - [Pandas](https://pandas.pydata.org/): Manipulação de dados.

---

## 📦 Como Executar o Projeto

1. **Clone o repositório**:
   ```bash
   git clone https://github.com/JackCaolho/Sonic_Fiap.git
