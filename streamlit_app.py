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

st.title("Agente de Exploração de Dados (LLM + CSV)")

# Inicializa o histórico na sessão
if "historico" not in st.session_state:
    st.session_state.historico = []

# 🔑 Pega a chave do secrets (configurada no Streamlit Cloud)
api_key = st.secrets["GROQ_API_KEY"]
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
    # Limpa a pasta files ao carregar um novo CSV
    utils.limpar_pasta_graficos(caminho)
    df = CSVAnalysisAgent.load_file(uploaded_file)
    if not isinstance(df, pd.DataFrame):  # Verificação mínima para falhas
        st.error("Falha ao carregar o CSV. Verifique o arquivo.")
    else:
        st.write("### Pré-visualização dos dados")
        st.dataframe(df.head())

    # Pergunta do usuário
    pergunta = st.text_area("❓ Faça uma pergunta sobre os dados:")

    if st.button("Perguntar", disabled=not pergunta):
        with st.spinner("Pensando..."):
            try:
                resposta_dict = CSVAnalysisAgent.analyze_csv(pergunta)
                resposta = resposta_dict["output"]
                
                # Armazena no histórico
                st.session_state.historico.append({"pergunta": pergunta, "resposta": resposta})
    
                st.success("Resposta do Agente:")
                st.write(resposta)

                # Verifica se a resposta é um caminho de arquivo ou ZIP
                if isinstance(resposta, str) and (resposta.endswith(".png") or resposta.endswith(".zip")):
                    # Resolve o caminho absoluto
                    file_path = os.path.abspath(resposta)
                    if os.path.exists(file_path):
                        file_name = os.path.basename(file_path)
                        with open(file_path, "rb") as f:
                            st.download_button(
                                label=f"📥 Baixar {file_name}",
                                data=f,
                                file_name=file_name,
                                mime="image/png" if resposta.endswith(".png") else "application/zip"
                            )
                    else:
                        st.warning(f"Arquivo {resposta} não encontrado.")
            except Exception as e:
                st.error(f"Erro ao processar: {e}")
    
# Exibe todo o histórico
st.subheader("Histórico:")
for idx, item in enumerate(st.session_state.historico, 1):
    st.markdown(f"**{idx}. Pergunta:** {item['pergunta']}")
    st.markdown(f"➡️ **Resposta:** {item['resposta']}")
    st.write("---")