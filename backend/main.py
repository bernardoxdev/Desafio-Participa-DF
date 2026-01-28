from libs.sheets import DataAPI

if __name__ == "__main__":
    """
    EXEMPLO DE USO PARA FAZER A ANALISE DOS DADOS
    A análise dos dados gera um documento csv que deve ser mostrado ao usuário, caso ele deseje pode marcar para gerar um xlsx
    """
    app = DataAPI()
    
    app.read_docs()
    
    df = app.get_data_frame(1)
    
    app.clear_text_col(df, 2)
    
    print(df.head())
    
    app.make_analise(df)