from pathlib import Path
# Importa Path, uma classe do Python usada para trabalhar com caminhos
# de arquivos e pastas de forma independente do sistema operacional.

import pandas as pd
# Importa a biblioteca pandas e dá a ela o apelido "pd".
# O pandas será usado para ler e manipular os CSVs.


ROOT = Path(__file__).resolve().parents[2]
# __file__ representa o caminho do arquivo Python atual.
# Path(__file__) transforma esse caminho em um objeto Path.
# .resolve() transforma o caminho em um caminho absoluto.
# .parents[2] sobe duas pastas a partir do arquivo atual.
#
# Exemplo aproximado:
# projeto/
# ├── data/
# ├── src/
# │   └── data/
# │       └── load_data.py  <- arquivo atual
#
# parents[0] -> src/data
# parents[1] -> src
# parents[2] -> projeto
#
# Portanto, ROOT passa a representar a raiz do projeto.


def load_data(use_processed=False, dataset=None):
    # Define uma função responsável por carregar os dados.
    #
    # use_processed=False:
    # por padrão, os dados carregados serão os dados brutos.
    #
    # dataset=None:
    # permite chamar a função explicitamente como:
    # load_data(dataset="raw")
    # ou
    # load_data(dataset="processed")


    """Load one consistent raw or processed dataset."""
    # Docstring da função.
    # É uma pequena documentação explicando que a função
    # carrega um conjunto de dados consistente:
    # ou todo raw ou todo processed.


    if dataset is not None:
        # Verifica se o parâmetro dataset foi informado.
        #
        # Exemplo:
        # load_data(dataset="processed")
        #
        # Nesse caso, dataset não será None.


        if dataset not in {"raw", "processed"}:
            # Verifica se o valor recebido é diferente de
            # "raw" e "processed".
            #
            # {"raw", "processed"} é um conjunto (set) contendo
            # os únicos valores aceitos.


            raise ValueError("dataset deve ser 'raw' ou 'processed'")
            # Interrompe a execução e gera um erro caso o usuário
            # tenha passado um dataset inválido.


        use_processed = dataset == "processed"
        # Converte a escolha do dataset para True/False.
        #
        # Se dataset == "processed":
        #     use_processed = True
        #
        # Se dataset == "raw":
        #     use_processed = False.


    # A escolha do dataset define os dois arquivos para evitar misturar fontes.
    # Comentário explicando que games e ratings devem vir do mesmo conjunto.
    # Isso evita, por exemplo, usar metadata raw junto com ratings processed.


    folder = ROOT / "data" / ("processed" if use_processed else "raw")
    # Monta o caminho da pasta que contém os dados.
    #
    # Se use_processed=True:
    #     ROOT/data/processed
    #
    # Se use_processed=False:
    #     ROOT/data/raw
    #
    # O operador / do Path é usado para juntar partes do caminho.


    games_name = "games_metadata_processed.csv" if use_processed else "games_metadata_5k.csv"
    # Escolhe o nome do arquivo de metadata.
    #
    # Processado:
    # games_metadata_processed.csv
    #
    # Raw:
    # games_metadata_5k.csv


    ratings_name = "ratings_processed.csv" if use_processed else "game_ratings.csv"
    # Escolhe o arquivo de avaliações.
    #
    # Processado:
    # ratings_processed.csv
    #
    # Raw:
    # game_ratings.csv


    games = pd.read_csv(folder / games_name)
    # Lê o CSV de jogos usando pandas.
    #
    # folder / games_name cria o caminho completo do arquivo.
    # O resultado é armazenado em um DataFrame chamado games.


    ratings = pd.read_csv(folder / ratings_name)
    # Lê o CSV de avaliações.
    # O resultado é armazenado no DataFrame ratings.


    ratings["user_id"] = ratings["user_id"].astype(str)
    # Converte todos os IDs dos usuários para string.
    #
    # Isso garante que "user_10" seja tratado como texto,
    # e evita inconsistências de tipo entre diferentes partes do projeto.


    ratings["game_id"] = pd.to_numeric(ratings["game_id"], errors="raise").astype(int)
    # Converte game_id para número.
    #
    # pd.to_numeric() tenta transformar os valores em números.
    # errors="raise" significa:
    # se existir algo que não possa ser convertido, gerar erro.
    #
    # Depois .astype(int) garante que o resultado seja inteiro.


    ratings["rating"] = pd.to_numeric(ratings["rating"], errors="raise")
    # Converte as avaliações para valores numéricos.
    #
    # Por exemplo:
    # "5" -> 5
    # "4" -> 4
    #
    # Se houver um valor inválido, errors="raise" gera erro.


    games["game_id"] = pd.to_numeric(games["game_id"], errors="raise").astype(int)
    # Faz a mesma conversão para game_id na tabela de jogos.
    #
    # Isso é importante porque games.game_id e ratings.game_id
    # precisam possuir o mesmo tipo para serem relacionados.


    return games, ratings
    # Retorna os dois DataFrames:
    #
    # games  -> informações dos jogos
    # ratings -> avaliações dos usuários