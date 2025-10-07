from langchain_groq.chat_models import ChatGroq
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.memory import ConversationBufferMemory
import pandas as pd

class CSVAnalysisAgent:
    def __init__(self, key: str):
        self.current_file = None
        self.df = None
        self.agent = None
        self.llm = self._init_llm(key)
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    def _init_llm(self, key: str):
        """Inicializa o LLM Groq."""
        return ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            api_key=key,
            base_url="https://api.groq.com"
        )

    def load_file(self, file):
        """Carrega o arquivo CSV e inicializa o agente."""
        try:
            self.df = self._read_csv(file)
            self.current_file = file if isinstance(file, str) else file.name
            prompt_inicial = self._get_prompt_inicial()
            self.agent = self._create_agent(prompt_inicial)
            return self.df
        except Exception:
            return False

    def _read_csv(self, file):
        """Lê o CSV de um caminho ou objeto UploadedFile."""
        if isinstance(file, str):
            return pd.read_csv(file)
        file.seek(0)
        return pd.read_csv(file)

    def _get_prompt_inicial(self) -> str:
        """Retorna o prompt inicial para o agente."""
        return """
            Você é um assistente especializado em análise de dados CSV. 
            Seu objetivo é fornecer análises detalhadas de EDA (Exploração de Dados) em formato de texto para perguntas simples ou gráficos, com no máximo um gráfico por coluna do arquivo.

            O que você deve fazer:
            1. Descrição dos Dados:
                - Identifique tipos de dados (numéricos, categóricos).
                - Informe distribuição de cada variável (histogramas, distribuições).
                - Informe intervalos (mínimo, máximo) e medidas de tendência central (média, mediana).
                - Informe variabilidade (desvio padrão, variância).
            2. Identificação de Padrões e Tendências:
                - Analise padrões ou tendências temporais.
                - Informe valores mais e menos frequentes.
                - Verifique agrupamentos (clusters) nos dados.
            3. Detecção de Outliers:
                - Detecte valores atípicos.
                - Analise impacto dos outliers.
                - Sugira estratégias de tratamento (remoção, transformação, investigação).
            4. Relações entre Variáveis:
                - Analise relações (gráficos de dispersão, tabelas cruzadas).
                - Informe correlações.
                - Indique variáveis com maior ou menor influência.
            5. Conclusões:
                - Sintetize conclusões com base nas análises.

            REGRAS:
            - Use matplotlib ou pandas plotting para gráficos, salvando em 'files'.
            - Use `python_repl_ast` para executar código, nunca diretamente.
            - Forneça respostas concisas e informativas.
            - Utilize memória para lembrar perguntas anteriores.
            - Use 'Final Answer' apenas para o caminho do arquivo (ex.: 'files/hist_V1.png') sem 'Action'.
            - Para gráficos, crie apenas o solicitado e retorne o caminho.
            Exemplo:
            ```python
            import matplotlib.pyplot as plt
            import os
            os.makedirs("files", exist_ok=True)
            plt.figure()
            df['V1'].hist(bins=30)
            plt.title("Distribuição de V1")
            plt.xlabel("V1")
            plt.ylabel("Frequência")
            plt.tight_layout()
            plt.savefig("files/hist_V1.png")
            plt.close()
            ```
            Histórico da conversa anterior: {chat_history}
            Pergunta atual: {input}
        """

    def _create_agent(self, prompt_inicial):
        """Cria o agente LangChain."""
        return create_pandas_dataframe_agent(
            self.llm,
            self.df,
            verbose=True,
            prefix=prompt_inicial,
            max_iterations=5000,
            agent_executor_kwargs={
                "memory": self.memory,
                "handle_parsing_errors": True
            },
            allow_dangerous_code=True
        )

    def analyze_csv(self, question: str) -> dict:
        """Analisa a pergunta usando o agente."""
        if not self.agent:
            return {"output": "Nenhum arquivo carregado."}
        try:
            result = self.agent.invoke(question)
            return {"output": result["output"] if isinstance(result, dict) and "output" in result else result}
        except Exception as e:
            return {"output": f"Erro ao processar a pergunta: {str(e)}"}