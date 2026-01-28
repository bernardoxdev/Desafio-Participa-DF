import os
import pandas as pd

from backend.libs.preprocessing import clean_text
from backend.libs.pii_detector import train_context_model, detect_pii, mask_text, analyze_text_multilabel

BASE_DIR = os.path.dirname(__file__)

class App():
    def __init__(self):
        self.DOCS = {}
        self.COLUNAS = {}
        self.TRATADOS_DIR = os.path.join(BASE_DIR, 'backend', 'data', 'processada')

    def clear_data(self) -> None:
        self.COLUNAS = {}

    def read_docs(self) -> dict:
        docs_dir = os.path.join(BASE_DIR, 'backend', 'data', 'raw')
        
        arquivos_existentes = set(self.DOCS.values())
        
        for filename in os.listdir(docs_dir):
            if filename not in arquivos_existentes:
                self.DOCS[len(self.DOCS) + 1] = filename
                    
        return self.DOCS

    def get_doc_path(self, doc_id: int) -> str:
        filename = self.DOCS.get(doc_id)
        
        if not filename:
            raise ValueError(f"Document with ID {doc_id} does not exist.")
        
        return os.path.join(BASE_DIR, 'backend', 'data', 'raw', filename)

    def get_data_frame(self, doc_id: int) -> pd.DataFrame:
        if not self.DOCS:
            self.read_docs()

        file_path = self.get_doc_path(doc_id)
        extensao = os.path.splitext(file_path)[1].lower()

        if extensao in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        elif extensao == '.json':
            df = pd.read_json(file_path)
        else:
            df = pd.read_csv(file_path)

        return df

    def get_df_columns(self, df: pd.DataFrame) -> list:
        dic = {}
        
        for item in df.columns.tolist():
            dic[len(dic) + 1] = item
        
        return dic
    
    def clear_text_col(self, df: pd.DataFrame, pos: int) -> pd.DataFrame:
        colunas = df.columns.tolist()
        
        if pos < 1 or pos > len(colunas):
            raise ValueError("Posição da coluna inválida.")

        col_name = colunas[pos - 1]
        new_col_name = f"{col_name}_clean"
        
        self.COLUNAS[len(self.COLUNAS)] = new_col_name

        df[new_col_name] = df[col_name].astype(str).apply(clean_text)

        return df
    
    def save_df(self, df: pd.DataFrame, name: str) -> None:
        df.to_csv(os.path.join(self.TRATADOS_DIR, name))

if __name__ == '__main__':
    app = App()
    
    app.read_docs()

    df = app.get_data_frame(1)
    
    print(app.get_df_columns(df))