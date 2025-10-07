import streamlit as st
import pandas as pd
import os
import tempfile
from zipfile import ZipFile
import matplotlib.pyplot as plt
from utils import Utils
from agent import CSVAnalysisAgent

def setup_app():
    """Configurações iniciais do app."""
    st.set_page_config(page_title="Agente CSV", layout="wide")
    st.title("Agente de Exploraçãos de Dados (LLM + CSV)")

def init_session_state():
    """Inicializa o estado da sessão."""
    if "historico" not in st.session_state:
        st.session_state.historico = []

def get_api_key() -> str:
    """Obtém a chave API do secrets."""
    api_key = st.secrets.get("GROQ_API_KEY", None)
    if not api_key:
        st.error("❌ Configure sua chave da Groq em st.secrets['GROQ_API_KEY']")
        st.stop()
    os.environ["GROQ_API_KEY"] = api_key
    return api_key

def init_agent_and_utils(api_key: str):
    """Inicializa o agente e utils."""
    return CSVAnalysisAgent(key=api_key), Utils()

def handle_upload(uploaded_file, utils, caminho: str = "files"):
    """Manipula o upload do CSV."""
    if uploaded_file:
        utils.limpar_pasta_graficos(caminho)
        df = CSVAnalysisAgent.load_file(uploaded_file)
        if not isinstance(df, pd.DataFrame):
            st.error("Falha ao carregar o CSV. Verifique o arquivo.")
            return None
        st.write("### Pré-visualização do CSV")
        st.dataframe(df.head())
        return df
    return None

def handle_question(pergunta: str, df):
    """Manipula a pergunta do usuário."""
    if st.button("Perguntar", disabled=not pergunta):
        with st.spinner("Pensando..."):
            try:
                resposta_dict = CSVAnalysisAgent.analyze_csv(pergunta)
                resposta = resposta_dict["output"]
                st.session_state.historico.append({"pergunta": pergunta, "resposta": resposta})
                st.success("Resposta do Agente:")
                st.write(resposta)
                show_download_if_file(resposta)
            except Exception as e:
                st.error(f"Erro ao processar: {e}")

def show_download_if_file(resposta: str):
    """Mostra botão de download se a resposta for um caminho de arquivo."""
    if isinstance(resposta, str) and (resposta.endswith(".png") or resposta.endswith(".zip")):
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

def display_history():
    """Exibe o histórico de perguntas e respostas."""
    st.subheader("Histórico de Perguntas e Respostas")
    for idx, item in enumerate(st.session_state.historico, 1):
        st.markdown(f"**{idx}. Pergunta:** {item['pergunta']}")
        st.markdown(f"➡️ **Resposta:** {item['resposta']}")
        st.write("---")

# Execução principal
setup_app()
init_session_state()
api_key = get_api_key()
CSVAnalysisAgent, utils = init_agent_and_utils(api_key)
uploaded_file = st.file_uploader("Carregue um arquivo CSV", type=["csv"])
df = handle_upload(uploaded_file, utils)
if df is not None:
    pergunta = st.text_area("❓ Faça uma pergunta sobre os dados:")
    handle_question(pergunta, df)
display_history()