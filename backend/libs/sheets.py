import os
import pandas as pd
import numpy as np

from libs.preprocessing import clean_text
from libs.pii_detector import train_context_model, load_context_model, process_text_row, train_class_model, load_class_model

BASE_DIR = os.path.dirname(__file__)

class DataAPI():
    def __init__(self):
        self.DOCS = {}
        self.COLUNAS = {}
        self.TRATADOS_DIR = os.path.join(BASE_DIR, '..', 'data', 'processada')

    def clear_data(self) -> None:
        self.COLUNAS = {}

    def read_docs(self) -> dict:
        docs_dir = os.path.join(BASE_DIR, '..', 'data', 'raw')
        
        arquivos_existentes = set(self.DOCS.values())
        
        for filename in os.listdir(docs_dir):
            if filename not in arquivos_existentes:
                self.DOCS[len(self.DOCS) + 1] = filename
                    
        return self.DOCS

    def get_doc_path(self, doc_id: int) -> str:
        filename = self.DOCS.get(doc_id)
        
        if not filename:
            raise ValueError(f"Document with ID {doc_id} does not exist.")
        
        return os.path.join(BASE_DIR, '..', 'data', 'raw', filename)

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
        new_col_name = "texto_clean"
        
        self.COLUNAS[len(self.COLUNAS)] = new_col_name

        df[new_col_name] = df[col_name].astype(str).apply(clean_text)

        return df
    
    def save_df(self, df: pd.DataFrame, name: str, csv: bool = True) -> None:
        base, ext = os.path.splitext(name)

        if not ext:
            ext = '.csv' if csv else '.xlsx'
            name = base + ext

        path = os.path.join(self.TRATADOS_DIR, name)

        if csv:
            df.to_csv(path, index=False)
        else:
            df.to_excel(path, index=False)
            
    def make_analise(self, df: pd.DataFrame, nome_documento_salvar: str = "documento_analisado", csv: bool = True, csv_train_context: list = None, csv_train_clas: list = None):
        try:
            context_model = load_context_model()
        except FileNotFoundError:
            if csv_train_context:
                context_model = train_context_model([os.path.join(BASE_DIR, "..", "data", "train", "dataset_context_1000.csv")].extend(csv_train_context))
            else:
                context_model = train_context_model([os.path.join(BASE_DIR, "..", "data", "train", "dataset_context_1000.csv")])
            
        try:
            clas_model = load_class_model()
        except FileNotFoundError:
            if csv_train_clas:
                clas_model = train_class_model([os.path.join(BASE_DIR, "..", "data", "train", "pii_context.csv")].extend(csv_train_clas))
            else:
                clas_model = train_class_model([os.path.join(BASE_DIR, "..", "data", "train", "pii_context.csv")])
        
        results = df['texto_clean'].apply(
            lambda x: process_text_row(x, context_model, clas_model)
        )
        
        results_df = pd.DataFrame(results.tolist())
        
        df = pd.concat([df, results_df], axis=1)
        
        cols = ['pii_regex', 'pii_brutils', 'pii_entities']
        df[cols] = df[cols].replace(r'^\s*$', np.nan, regex=True)
        df = df.dropna(subset=cols, how='all')
        
        df = df.reset_index(drop=True)
        
        self.save_df(df, nome_documento_salvar, csv)
        
if __name__ == '__main__':
    pass