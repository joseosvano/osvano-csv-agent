from langchain_groq.chat_models import ChatGroq
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import AIMessage, HumanMessage
import pandas as pd

class CSVAnalysisAgent:
    def __init__(self, key: str):
        self.current_file = None
        self.df = None
        self.agent = None

        # Inicializando ChatOpenAI para Groq
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",                    # modelo Groq
            temperature=0,
            api_key=key,
            base_url="https://api.groq.com"
        )

        self.llm_open = ChatOpenAI(
            model="gpt-4.1-nano",
            temperature=0,
            api_key=key,
            base_url="https://api.openai.com/v1"
        )
        
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", 
            return_messages=True
        )

    def load_file(self, file):
        try:
            # Lida com string (path) ou UploadedFile (objeto de arquivo)
            if isinstance(file, str):
                self.df = pd.read_csv(file)
            else:  # Assume UploadedFile do Streamlit
                file.seek(0)  # Reseta o ponteiro do arquivo (boa prática)
                self.df = pd.read_csv(file)
            self.current_file = file if isinstance(file, str) else file.name            
            prompt_inicial = """
                Você é um assistente especializado em análise de dados CSV. 
                Seu objetivo é permitir que o usuário faça perguntas sobre qualquer arquivo CSV carregado e fornecer análises detalhadas de EDA (Exploração de Dados)
                em formato de texto caso a pergunta seja simples ou em graficos e a quantidade de gráficos deve ser de no máximo o numero de colunas do arquivo carregado.

                O que você deve fazer:
                1. Descrição dos Dados:
                    - Identifique tipos de dados (numéricos, categóricos).
                    - Informe distribuição de cada variável (histogramas, distribuições).
                    - Informe intervalos (mínimo, máximo) e medidas de tendência central (média, mediana).
                    - Informe variabilidade (desvio padrão, variância).
                2. Identificação de Padrões e Tendências:
                    - Analise padrões ou tendências temporais.
                    - Informe os valores mais frequentes e menos frequentes.
                    - Verifique a existência de agrupamentos (clusters) nos dados.
                3. Detecção de Outliers:
                    - Detecte valores atípicos.
                    - Analise como os outliers afetam a análise.
                    - Sugira estratégias de tratamento (remoção, transformação, investigação).
                4. Relações entre Variáveis:
                    - Analise relações entre variáveis (gráficos de dispersão, tabelas cruzadas).
                    - Informe correlações.
                    - Indique variáveis com maior ou menor influência sobre outras.
                5. Conclusões:
                    - Sempre sintetize conclusões com base nas análises realizadas.

                    REGRAS:
                    
                    - Para análise gráfica, utilize matplotlib ou pandas plotting e retorne o codigo para execução em formato de listas para mais de um grafico.
                    - Nunca execute código diretamente na conversa; use sempre a ferramenta `python_repl_ast`.
                    - Sempre tente fornecer respostas concisas e informativas.
                    - Utilize a memória para lembrar perguntas anteriores e sintetizar conclusões ao longo da análise.
                    - Só use "Final Answer" quando NÃO houver mais "Action".
                    - Nunca coloque "Final Answer" junto com um "Action".
                    - Se você gerar código Python, NÃO escreva a conclusão até que ele seja executado.
                    - Retorne Thought junto com a resposta final

                    IMPORTANTE:
                    - Nunca escreva "Final Answer" junto com "Action".
                    - Sempre gere código Python primeiro e use Action.
                    - Depois que o código for executado, use Final Answer apenas para retornar o caminho do(s) arquivo(s).
                    - Em caso de solicitação de gráficos na pergunta, CRIE SOMENTE O GRAFICO SOLICITADO e retorne somente o caminho ("files") do arquivo ou ZIP e nunca reimprima os gráficos caso
                    contrário NÃO gerar graficos retornar somente a resposta. 
                    Exemplo:
                    ```python
                    # código Python válido 
                    import matplotlib.pyplot as plt
                    import os
                    from zipfile import ZipFile

                    os.makedirs("files", exist_ok=True)
                    arquivos = []

                    numerical_cols = df.select_dtypes(include='number').columns

                    for col in numerical_cols:
                        caminho = f"files/hist_{{col}}.png"
                        plt.figure()
                        df[col].hist(bins=30)
                        plt.title(f"Distribuição de {{col}}")
                        plt.xlabel(col)
                        plt.ylabel("Frequência")
                        plt.tight_layout()
                        plt.savefig(caminho)
                        plt.close()
                        arquivos.append(caminho)

                    if len(arquivos) > 1:
                        with ZipFile("files/graficos.zip", "w") as zipf:
                            for f in arquivos:
                                zipf.write(f, os.path.basename(f))
                    ``` 
                    Histórico da conversa anterior: {chat_history}
                    Pergunta atual: {input}
            """

            self.agent = create_pandas_dataframe_agent(
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
            return self.df
        except Exception as e:
            print("Erro ao carregar CSV:", e)
            return False

    def analyze_csv(self, question: str):
        if not self.agent:
            return {"output": "Nenhum arquivo carregado."}
        try:
            result = self.agent.invoke(question)
            return {"output": result}
        except Exception as e:
            return {"output": f"Erro ao processar a pergunta: {str(e)}"}
