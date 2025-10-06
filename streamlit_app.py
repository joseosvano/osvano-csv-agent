import streamlit as st
import pandas as pd
import os
import tempfile
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_groq.chat_models import ChatGroq
from langchain.memory import ConversationBufferMemory
from zipfile import ZipFile
import matplotlib.pyplot as plt
from utils import Utils
from agent import CSVAnalysisAgent

st.set_page_config(page_title="Agente CSV LLM", layout="wide")

st.title("🤖 Agente de Análise Exploratória de Dados (LLM + CSV)")

# Inicializa o histórico na sessão
if "historico" not in st.session_state:
    st.session_state.historico = []

# 🔑 Pega a chave do secrets (configurada no Streamlit Cloud)
api_key =  st.secrets["GROQ_API_KEY"]
if "GROQ_API_KEY" not in st.secrets:
    st.error("❌ Configure sua chave da OpenAI em st.secrets['GROQ_API_KEY']")
else:
    os.environ["GROQ_API_KEY"] = api_key

CSVAnalysisAgent = CSVAnalysisAgent(key=api_key)
utils = Utils()

# 📂 Upload CSV
uploaded_file = st.file_uploader("Carregue um arquivo CSV", type=["csv"])
caminho = "files"
if uploaded_file and api_key:
    df = CSVAnalysisAgent.load_file(uploaded_file)
    df = pd.read_csv(uploaded_file)

    st.write("### Pré-visualização dos dados")
    st.dataframe(df.head())

    # Pergunta do usuário
    pergunta = st.text_area("❓ Faça uma pergunta sobre os dados:")

    if st.button("Perguntar", disabled=not pergunta):
        with st.spinner("Pensando..."):
            try:
                resposta = CSVAnalysisAgent.agent.run(pergunta)
                
                # Armazena no histórico
                st.session_state.historico.append({"pergunta": pergunta, "resposta": resposta})
    
                st.success("Resposta do Agente:")
                st.write(resposta)
            except Exception as e:
                st.error(f"Erro ao processar: {e}")
    
    # ⚡ Extra: gerar ZIP com histogramas
    st.write("### Gerar histogramas de todas as colunas numéricas")
    if st.button("Gerar ZIP de gráficos", disabled=utils.verificar_pasta_arquivos("files")):
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if not num_cols:
            st.warning("Nenhuma coluna numérica encontrada!")
        else:
            with tempfile.TemporaryDirectory() as tmpdir:
                arquivos = []
                for col in num_cols:
                    fig, ax = plt.subplots()
                    df[col].hist(bins=30, ax=ax)
                    ax.set_title(f"Distribuição de {col}")
                    caminho = os.path.join(tmpdir, f"{col}.png")
                    fig.savefig(caminho)
                    arquivos.append(caminho)
                    plt.close(fig)

                zip_path = os.path.join(tmpdir, "graficos.zip")
                with ZipFile(zip_path, "w") as zipf:
                    for f in arquivos:
                        zipf.write(f, os.path.basename(f))

                with open(zip_path, "rb") as f:
                    st.download_button("📥 Baixar gráficos ZIP", f, file_name="graficos.zip")

# Exibe todo o histórico
st.subheader("Histórico de Perguntas e Respostas")
for idx, item in enumerate(st.session_state.historico, 1):
    st.markdown(f"**{idx}. Pergunta:** {item['pergunta']}")
    st.markdown(f"➡️ **Resposta:** {item['resposta']}")
    st.write("---")
