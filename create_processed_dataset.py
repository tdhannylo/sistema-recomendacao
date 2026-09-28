from pathlib import Path
# Será usado principalmente para criar uma matriz esparsa com valores 1 indicando a existência de uma avaliação.
import numpy as np
import pandas as pd
# Importa csr_matrix, uma estrutura para representar matrizes esparsas
from scipy.sparse import csr_matrix
from src.data.preprocess import validate_data

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
# Define que serão selecionados usuários que estejam acima ou no percentil 80% de quantidade de avaliações
USER_PERCENTILE = 0.80
GAME_PERCENTILE = 0.80

def load_raw_data():
    games = pd.read_csv(RAW_DIR / "games_metadata_5k.csv")
    ratings = pd.read_csv(RAW_DIR / "game_ratings.csv")
    ratings["user_id"] = ratings["user_id"].astype(str)
    # errors="raise" faz o programa falhar caso exista um valor que não possa ser convertido.
    ratings["game_id"] = pd.to_numeric(ratings["game_id"], errors="raise").astype(int)
    ratings["rating"] = pd.to_numeric(ratings["rating"], errors="raise")
    games["game_id"] = pd.to_numeric(games["game_id"], errors="raise").astype(int)
    return games, ratings

# Recebe a tabela de avaliações e calcula a distribuição de quantidade de jogos em comum entre usuários
def overlap_distribution(ratings):
    # Obtém todos os usuários únicos
    users = pd.Index(ratings["user_id"].unique())
    games = pd.Index(ratings["game_id"].unique())
    # Converte cada user_id para uma posição numérica
    user_codes = users.get_indexer(ratings["user_id"])
    game_codes = games.get_indexer(ratings["game_id"])

    # Cria uma matriz esparsa usuário × jogo para saber se o usuário avaliou o jogo
    matrix = csr_matrix(
        (np.ones(len(ratings), dtype=np.int8), (user_codes, game_codes)),
        shape=(len(users), len(games)),
    )

    # Multiplica a matriz pela sua transposta, produto é quant de jogos em comum
    overlap = (matrix @ matrix.T).tocoo()
    # Seleciona somente uma metade da matriz
    values = overlap.data[overlap.row < overlap.col]
    # Conta quantos pares possuem 1 jogo em comum ou 2 ou 3 e transforma em dicionário
    distribution = pd.Series(values).value_counts().sort_index().to_dict()
    # Calcula quantos pares diferentes de usuários existem
    total_pairs = len(users) * (len(users) - 1) // 2
    # Descobre quantos pares possuem ZERO jogos em comum
    distribution[0] = total_pairs - len(values)
    # Descobre quantos pares possuem ZERO jogos em comum e garante que chaves e valores sejam inteiros
    return dict(sorted((int(key), int(value)) for key, value in distribution.items()))

# Calcula estatísticas gerais de um conjunto de ratings
def dataset_stats(ratings):
    # Conta quantos usuários diferentes existem
    user_count = ratings["user_id"].nunique()
    game_count = ratings["game_id"].nunique()
     # Conta quantas avaliações existem
    rating_count = len(ratings)
    # Calcula a densidade da matriz usuário × jogo
    density = rating_count / (user_count * game_count)
    
    # Retorna todas as estatísticas em um dicionário
    return {
        # Número de usuários.
        "users": user_count,
        "games": game_count,
        "ratings": rating_count,
        "density": density,
        "sparsity": 1 - density,
        # Conta quantas avaliações cada usuário possui, depois calcula a média
        "mean_ratings_user": ratings.groupby("user_id").size().mean(),
        "mean_ratings_game": ratings.groupby("game_id").size().mean(),
        # Calcula a distribuição de jogos em comum entre usuários
        "overlap": overlap_distribution(ratings), 
    }


def create_processed_dataset():
    games, ratings = load_raw_data()
    # Verifica se os dados originais são válidos.
    validate_data(games, ratings)

    # Conta quantas avaliações cada usuário possui
    user_counts = ratings.groupby("user_id").size()
    # Conta quantas avaliações cada jogo possui
    game_counts = ratings.groupby("game_id").size()
    # Calcula o percentil 80% da quantidade de avaliações por usuário
    user_threshold = user_counts.quantile(USER_PERCENTILE)
    game_threshold = game_counts.quantile(GAME_PERCENTILE)
    # Seleciona usuários cuja quantidade de avaliações seja igual ou superior ao limite
    selected_users = user_counts[user_counts >= user_threshold].index
    # Seleciona jogos cuja quantidade de avaliações seja igual ou superior ao limite
    selected_games = game_counts[game_counts >= game_threshold].index
    
    # Mantém somente avaliações que satisfazem AMBAS as condições: usuário foi selecionado e jogo foi selecionado
    processed_ratings = ratings[
        ratings["user_id"].isin(selected_users)
        & ratings["game_id"].isin(selected_games)
    ].copy()

    # Mantém na tabela de jogos somente os jogos que realmente aparecem nas avaliações processadas
    processed_games = games[games["game_id"].isin(processed_ratings["game_id"])].copy()

    # Ordena as avaliações por usuário e jogo.
    processed_ratings = processed_ratings.sort_values(
        ["user_id", "game_id"]
    ).reset_index(drop=True)

    # Ordena os jogos pelo ID e recria o índice
    processed_games = processed_games.sort_values("game_id").reset_index(drop=True)
    # Valida novamente o dataset depois do processamento
    validate_data(processed_games, processed_ratings)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    processed_games.to_csv(
        PROCESSED_DIR / "games_metadata_processed.csv",
        index=False
    )
    processed_ratings.to_csv(
        PROCESSED_DIR / "ratings_processed.csv",
        index=False
    )

    # Retorna dados originais, dados, limite de usuários e jogos
    return games, ratings, processed_games, processed_ratings, user_threshold, game_threshold
    
# Exibe estatísticas formatadas no terminal
def print_stats(label, ratings):
    # Calcula as estatísticas
    stats = dataset_stats(ratings)
    # Imprime o nome do dataset
    print(f"\n{label}")
    # Imprime quantidade de usuários, jogos e avaliações
    print(f"usuarios={stats['users']} jogos={stats['games']} ratings={stats['ratings']}")
    print(f"densidade={stats['density']:.6f} esparsidade={stats['sparsity']:.6f}")
    print(f"media ratings/usuario={stats['mean_ratings_user']:.2f} "
          f"media ratings/jogo={stats['mean_ratings_game']:.2f}")

    # Mostra quantos pares de usuários possuem 0, 1, 2 ou 3 ou mais jogos em comum
    print(f"overlap 0={stats['overlap'].get(0, 0)} "
          f"1={stats['overlap'].get(1, 0)} 2={stats['overlap'].get(2, 0)} "
          f"3+={sum(value for key, value in stats['overlap'].items() if key >= 3)}")


if __name__ == "__main__":
    # Executa todo o processo de criação do dataset processado
    raw_games, raw_ratings, processed_games, processed_ratings, user_threshold, game_threshold = create_processed_dataset()
    print(f"Criterio reproduzivel: usuarios >= quantil {USER_PERCENTILE:.0%} ({user_threshold:.0f}) "
          f"e jogos >= quantil {GAME_PERCENTILE:.0%} ({game_threshold:.0f})")
    print("nao houve amostragem aleatoria nem ratings inventados")
    print_stats("DATASET ORIGINAL (data/raw)", raw_ratings)
    print_stats("DATASET PROCESSADO (data/processed)", processed_ratings)
    print("Arquivos gerados: data/processed/games_metadata_processed.csv, "
          "data/processed/ratings_processed.csv")