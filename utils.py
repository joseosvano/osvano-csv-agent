
import os
import glob


class Utils:
    
    @staticmethod
    def limpar_pasta_graficos(pasta: str):
        print("limpar_pasta_graficos")
        """
        Remove todos os arquivos PNG e ZIP da pasta 'files' antes de gerar novos gráficos.
        """
        
        # Certifica-se que a pasta existe
        os.makedirs(pasta, exist_ok=True)

        # Buscar todos os PNG e ZIP
        arquivos = glob.glob(os.path.join(pasta, "*.png")) + glob.glob(os.path.join(pasta, "*.zip"))
        print("arquivos")
        print(arquivos)
        for arq in arquivos:
            try:
                os.remove(arq)
                print(f"Removido: {arq}")
            except Exception as e:
                print(f"Erro ao remover {arq}: {e}")

    @staticmethod
    def verificar_pasta_arquivos(pasta: str):
        print("verificar_pasta_arquivos")
        """
        Verifica se a pasta 'files' contém arquivos PNG ou ZIP.
        """
        arquivos = glob.glob(os.path.join(pasta, "*.png")) + glob.glob(os.path.join(pasta, "*.zip"))
        if arquivos:
            print(f"Arquivos encontrados em {pasta}: {arquivos}")
            return False
        else:
            print(f"Nenhum arquivo encontrado em {pasta}.")
            return True
