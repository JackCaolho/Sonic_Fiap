import os
import whisper
import pandas as pd
import soundfile as sf
import librosa
import streamlit as st
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

# Baixar os recursos necessários do nltk
nltk.download('stopwords')
nltk.download('vader_lexicon')

nlp = spacy.load("pt_core_news_sm")

def tratar_audio(audio_bytes, filename):
    y, sr_rate = librosa.load(audio_bytes, sr=16000)
    caminho_temp_tratado = f"temp_tratado_{filename}.wav"
    sf.write(caminho_temp_tratado, y, sr_rate)
    return caminho_temp_tratado

def transcrever_audio(caminho_arquivo):
    model = whisper.load_model("small")
    resultado = model.transcribe(caminho_arquivo, language="pt")
    return resultado['text']

def preparar_texto_para_analise_spacy(texto):
    doc = nlp(texto.lower())
    tokens_filtrados = [token.text for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens_filtrados)

def analisar_sentimentos(texto):
    sia = SentimentIntensityAnalyzer()
    return sia.polarity_scores(texto)

def separar_frases(transcricao):
    frases_funcionario = []
    frases_cliente = []
    frases = transcricao.split('. ')
    
    for frase in frases:
        if "nota" in frase.lower() and "de 0 a 10" in frase.lower():
            frases_funcionario.append(frase)
        else:
            frases_cliente.append(frase)
    
    return frases_funcionario, frases_cliente

def exibir_resultados_sentimento(sentimentos):
    st.write(f"**Negatividade:** {sentimentos['neg']:.2f}")
    st.write(f"**Neutralidade:** {sentimentos['neu']:.2f}")
    st.write(f"**Positividade:** {sentimentos['pos']:.2f}")
    st.write(f"**Pontuação Composta:** {sentimentos['compound']:.2f}")

    if sentimentos['compound'] >= 0.05:
        st.write("O sentimento geral do texto é positivo.")
    elif sentimentos['compound'] <= -0.05:
        st.write("O sentimento geral do texto é negativo.")
    else:
        st.write("O sentimento geral do texto é neutro.")

    # Criar gráfico de barras horizontais para visualização dos sentimentos
    labels = ['Negatividade', 'Neutralidade', 'Positividade']
    sizes = [sentimentos['neg'], sentimentos['neu'], sentimentos['pos']]
    colors = ['#FF6F6F', '#6F6F6F', '#6FFF6F']

    fig, ax = plt.subplots()
    ax.barh(labels, sizes, color=colors)
    ax.set_xlabel('Proporção')
    ax.set_title('Distribuição dos Sentimentos')

    st.pyplot(fig)

# Configuração do Streamlit
st.title("Transcrição e Análise de Sentimentos de Áudio")
st.write("Faça o upload de um arquivo de áudio para transcrição e tratamento.")

# Upload de áudio
audio_file = st.file_uploader("Escolha um arquivo de áudio", type=["wav", "mp3"])

if audio_file is not None:
    # Exibir nome do arquivo
    st.audio(audio_file)
    filename = audio_file.name
    
    # Tratar o áudio (ajuste de taxa de amostragem)
    with open(filename, "wb") as f:
        f.write(audio_file.getbuffer())
    
    caminho_tratado = tratar_audio(audio_file, filename)
    
    # Transcrever o áudio
    st.write("Transcrevendo o áudio...")
    transcricao = transcrever_audio(caminho_tratado)
    
    # Exibir a transcrição
    st.write("**Transcrição original:**")
    st.text(transcricao)
    
    # Separar as frases ditas pelo funcionário e pelo cliente
    frases_funcionario, frases_cliente = separar_frases(transcricao)
    
    # Preparar o texto dos clientes para análise
    texto_clientes = " ".join(frases_cliente)
    texto_preparado = preparar_texto_para_analise_spacy(texto_clientes)
    st.write("**Texto tratado para análise de sentimentos:**")
    st.text(texto_preparado)

    # Análise de sentimentos
    st.write("**Análise de sentimentos:**")
    sentimentos = analisar_sentimentos(texto_preparado)
    exibir_resultados_sentimento(sentimentos)
    
    # Oferecer o download do áudio tratado
    st.write("Áudio tratado:")
    with open(caminho_tratado, "rb") as f:
        audio_bytes_tratado = f.read()
    st.audio(audio_bytes_tratado)
    
    # Botão para baixar o áudio tratado
    st.download_button("Baixar áudio tratado", data=audio_bytes_tratado, file_name=f"tratado_{filename}")
    
    # Botão para baixar a análise de sentimentos em CSV
    df = pd.DataFrame([{"Nome do Áudio": filename, "Texto Tratado": texto_preparado, "Sentimentos": sentimentos}])
    st.download_button("Baixar análise de sentimentos", data=df.to_csv(index=False), file_name="analise_sentimentos.csv")