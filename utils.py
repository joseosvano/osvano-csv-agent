
import os
import glob

class Utils:
    @staticmethod
    def limpar_pasta_graficos(pasta: str):
        """
        Remove todos os arquivos PNG e ZIP da pasta especificada.
        """
        os.makedirs(pasta, exist_ok=True)
        arquivos = glob.glob(os.path.join(pasta, "*.png")) + glob.glob(os.path.join(pasta, "*.zip"))
        for arq in arquivos:
            try:
                os.remove(arq)
            except Exception:
                pass  # Silencia erros para evitar interrupções

    @staticmethod
    def verificar_pasta_arquivos(pasta: str) -> bool:
        """
        Verifica se a pasta contém arquivos PNG ou ZIP.
        Retorna True se vazia, False caso contrário.
        """
        arquivos = glob.glob(os.path.join(pasta, "*.png")) + glob.glob(os.path.join(pasta, "*.zip"))
        return len(arquivos) == 0