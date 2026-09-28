# Classe de Python usada para trabalhar com caminhos de arquivos
from pathlib import Path
# O pandas será usado para ler e manipular os CSVs.
import pandas as pd


# Irá pegar o arquivo de load e ver onde está a pasta atual com o dataset usando parents
ROOT = Path(__file__).resolve().parents[2]

# Define uma função responsável por carregar os dados, aqui será carregado ou raw ou processed
def load_data(use_processed=False, dataset=None):
    # Verifica se o dataset foi informado
    if dataset is not None:
        # Verifica se o valor recebido é diferente de "raw" e "processed".
        if dataset not in {"raw", "processed"}:
            raise ValueError("dataset deve ser 'raw' ou 'processed'")
        # Converte a escolha do dataset para True/False.
        use_processed = dataset == "processed"

    folder = ROOT / "data" / ("processed" if use_processed else "raw")
    games_name = "games_metadata_processed.csv" if use_processed else "games_metadata_5k.csv"
    ratings_name = "ratings_processed.csv" if use_processed else "game_ratings.csv"
    games = pd.read_csv(folder / games_name)
    ratings = pd.read_csv(folder / ratings_name)

    # Converte todos os IDs dos usuários para string.
    ratings["user_id"] = ratings["user_id"].astype(str)

    # Converte game_id para número, valor inválido caso não for inteiro
    ratings["game_id"] = pd.to_numeric(ratings["game_id"], errors="raise").astype(int)
    
    ratings["rating"] = pd.to_numeric(ratings["rating"], errors="raise")

    # games.game_id e ratings.game_id precisam possuir o mesmo tipo para serem relacionados
    games["game_id"] = pd.to_numeric(games["game_id"], errors="raise").astype(int)
    return games, ratings