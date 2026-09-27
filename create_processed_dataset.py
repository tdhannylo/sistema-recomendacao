from pathlib import Path
# Importa Path para trabalhar com caminhos de arquivos e diretórios.

import numpy as np
# Importa NumPy.
# Será usado principalmente para criar uma matriz esparsa
# com valores 1 indicando a existência de uma avaliação.

import pandas as pd
# Importa pandas para leitura, filtragem e análise dos CSVs.

from scipy.sparse import csr_matrix
# Importa csr_matrix, uma estrutura para representar matrizes esparsas
# de maneira mais eficiente em memória.

from src.data.preprocess import validate_data
# Importa a função que verifica se os dados possuem
# estrutura e valores válidos.


ROOT = Path(__file__).resolve().parent
# Obtém a pasta onde este arquivo Python está localizado.
#
# Diferentemente do ROOT usado anteriormente:
# .parent sobe apenas uma pasta.
#
# Se o arquivo estiver em:
# projeto/create_processed_dataset.py
#
# ROOT será:
# projeto/


RAW_DIR = ROOT / "data" / "raw"
# Cria o caminho para:
# projeto/data/raw


PROCESSED_DIR = ROOT / "data" / "processed"
# Cria o caminho para:
# projeto/data/processed


USER_PERCENTILE = 0.80
# Define que serão selecionados usuários que estejam
# acima ou no percentil 80% de quantidade de avaliações.


GAME_PERCENTILE = 0.80
# Define o mesmo critério para os jogos.
#
# Serão considerados os jogos com quantidade de avaliações
# igual ou superior ao percentil 80%.


def load_raw_data():
    # Define uma função responsável por carregar
    # exclusivamente os dados originais.


    games = pd.read_csv(RAW_DIR / "games_metadata_5k.csv")
    # Lê o arquivo original de metadata dos jogos.


    ratings = pd.read_csv(RAW_DIR / "game_ratings.csv")
    # Lê o arquivo original de avaliações.


    ratings["user_id"] = ratings["user_id"].astype(str)
    # Garante que os IDs dos usuários sejam strings.


    ratings["game_id"] = pd.to_numeric(ratings["game_id"], errors="raise").astype(int)
    # Converte os IDs dos jogos para inteiros.
    #
    # errors="raise" faz o programa falhar caso exista
    # um valor que não possa ser convertido.


    ratings["rating"] = pd.to_numeric(ratings["rating"], errors="raise")
    # Converte as avaliações para números.


    games["game_id"] = pd.to_numeric(games["game_id"], errors="raise").astype(int)
    # Garante que os game_ids da tabela de jogos também sejam inteiros.


    return games, ratings
    # Retorna:
    #
    # games  -> metadata dos jogos
    # ratings -> avaliações originais

def overlap_distribution(ratings):
    # Recebe a tabela de avaliações e calcula
    # a distribuição de quantidade de jogos em comum entre usuários.


    users = pd.Index(ratings["user_id"].unique())
    # Obtém todos os usuários únicos.
    #
    # pd.Index é uma estrutura do pandas adequada para trabalhar
    # com os índices da matriz.


    games = pd.Index(ratings["game_id"].unique())
    # Obtém todos os jogos únicos.


    user_codes = users.get_indexer(ratings["user_id"])
    # Converte cada user_id para uma posição numérica.
    #
    # Exemplo:
    #
    # users:
    # [user_1, user_2, user_3]
    #
    # pode virar:
    # [0, 0, 1, 2, 1, ...]


    game_codes = games.get_indexer(ratings["game_id"])
    # Faz a mesma conversão para game_id.


    matrix = csr_matrix(
        (np.ones(len(ratings), dtype=np.int8), (user_codes, game_codes)),
        shape=(len(users), len(games)),
    )
    # Cria uma matriz esparsa usuário × jogo.
    #
    # Cada avaliação existente recebe o valor 1.
    #
    # Não importa aqui se a avaliação foi 1, 3 ou 5.
    #
    # Queremos apenas saber:
    # "este usuário avaliou este jogo?"
    #
    # np.ones(len(ratings))
    # cria uma sequência de 1s com uma posição para cada rating.
    #
    # (user_codes, game_codes)
    # indica em quais posições esses 1s devem ficar.
    #
    # shape define:
    # quantidade de usuários × quantidade de jogos.


    overlap = (matrix @ matrix.T).tocoo()
    # Multiplica a matriz pela sua transposta.
    #
    # matrix:
    #       jogos
    # user1  1 1 0
    # user2  1 0 1
    #
    # matrix.T:
    #       users
    #
    # O produto entre elas informa quantos jogos
    # cada par de usuários possui em comum.
    #
    # .tocoo() converte o resultado para o formato COO,
    # que facilita acessar linha, coluna e valor dos elementos.


    values = overlap.data[overlap.row < overlap.col]
    # Seleciona somente uma metade da matriz.
    #
    # Como a similaridade entre A e B é igual à de B e A,
    # não precisamos contar os dois pares.
    #
    # row < col elimina:
    # A-A
    # B-A quando A-B já foi considerado
    #
    # e mantém apenas:
    # A-B
    # A-C
    # B-C


    distribution = pd.Series(values).value_counts().sort_index().to_dict()
    # Conta quantos pares possuem:
    #
    # 1 jogo em comum
    # 2 jogos em comum
    # 3 jogos em comum
    # etc.
    #
    # Depois transforma o resultado em dicionário.


    total_pairs = len(users) * (len(users) - 1) // 2
    # Calcula quantos pares diferentes de usuários existem.
    #
    # Fórmula:
    # n × (n - 1) / 2
    #
    # Exemplo com 4 usuários:
    # 4 × 3 / 2 = 6 pares.


    distribution[0] = total_pairs - len(values)
    # Descobre quantos pares possuem ZERO jogos em comum.
    #
    # total_pairs = todos os pares possíveis
    # len(values) = pares que possuem pelo menos um jogo em comum


    return dict(sorted((int(key), int(value)) for key, value in distribution.items()))
    # Ordena o dicionário pela quantidade de jogos em comum
    # e garante que chaves e valores sejam inteiros.


def dataset_stats(ratings):
    # Calcula estatísticas gerais de um conjunto de ratings.


    user_count = ratings["user_id"].nunique()
    # Conta quantos usuários diferentes existem.


    game_count = ratings["game_id"].nunique()
    # Conta quantos jogos diferentes existem.


    rating_count = len(ratings)
    # Conta quantas avaliações existem.


    density = rating_count / (user_count * game_count)
    # Calcula a densidade da matriz usuário × jogo.
    #
    # Fórmula:
    #
    # avaliações existentes
    # ---------------------
    # combinações possíveis
    #
    # Se existem muitos espaços vazios,
    # a densidade será baixa.


    return {
        # Retorna todas as estatísticas em um dicionário.


        "users": user_count,
        # Número de usuários.


        "games": game_count,
        # Número de jogos.


        "ratings": rating_count,
        # Número de avaliações.


        "density": density,
        # Densidade.


        "sparsity": 1 - density,
        # Esparsidade.
        #
        # Se densidade = 0.01,
        # sparsidade = 0.99.


        "mean_ratings_user": ratings.groupby("user_id").size().mean(),
        # Agrupa por usuário.
        # Conta quantas avaliações cada usuário possui.
        # Depois calcula a média.


        "mean_ratings_game": ratings.groupby("game_id").size().mean(),
        # Faz o mesmo para os jogos:
        # média de avaliações por jogo.


        "overlap": overlap_distribution(ratings),
        # Calcula a distribuição de jogos em comum entre usuários.
    }


def create_processed_dataset():
    # Cria o dataset processado.


    games, ratings = load_raw_data()
    # Carrega os dados originais.


    validate_data(games, ratings)
    # Verifica se os dados originais são válidos.
    #
    # Se houver problema, a execução é interrompida.


    # O processado seleciona entidades densas, mas preserva apenas ratings reais.
    # Ou seja:
    # não são criadas avaliações artificiais.


    user_counts = ratings.groupby("user_id").size()
    # Conta quantas avaliações cada usuário possui.


    game_counts = ratings.groupby("game_id").size()
    # Conta quantas avaliações cada jogo possui.


    user_threshold = user_counts.quantile(USER_PERCENTILE)
    # Calcula o percentil 80% da quantidade de avaliações por usuário.
    #
    # Esse valor será usado como limite.


    game_threshold = game_counts.quantile(GAME_PERCENTILE)
    # Calcula o percentil 80% para os jogos.


    selected_users = user_counts[user_counts >= user_threshold].index
    # Seleciona usuários cuja quantidade de avaliações
    # seja igual ou superior ao limite.


    selected_games = game_counts[game_counts >= game_threshold].index
    # Seleciona jogos cuja quantidade de avaliações
    # seja igual ou superior ao limite.


    processed_ratings = ratings[
        ratings["user_id"].isin(selected_users)
        & ratings["game_id"].isin(selected_games)
    ].copy()
    # Mantém somente avaliações que satisfazem AMBAS as condições:
    #
    # 1. usuário foi selecionado
    # 2. jogo foi selecionado
    #
    # Importante:
    # as avaliações continuam sendo as avaliações originais.


    processed_games = games[games["game_id"].isin(processed_ratings["game_id"])].copy()
    # Mantém na tabela de jogos somente os jogos
    # que realmente aparecem nas avaliações processadas.


    processed_ratings = processed_ratings.sort_values(
        ["user_id", "game_id"]
    ).reset_index(drop=True)
    # Ordena as avaliações por usuário e jogo.
    #
    # reset_index(drop=True) recria o índice:
    # 0, 1, 2, 3...


    processed_games = processed_games.sort_values("game_id").reset_index(drop=True)
    # Ordena os jogos pelo ID e recria o índice.


    validate_data(processed_games, processed_ratings)
    # Valida novamente o dataset depois do processamento.
    #
    # Isso garante que o processamento não produziu
    # um conjunto inconsistente.


    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    # Cria a pasta data/processed caso ela ainda não exista.
    #
    # parents=True:
    # permite criar pastas intermediárias.
    #
    # exist_ok=True:
    # não gera erro se a pasta já existir.


    processed_games.to_csv(
        PROCESSED_DIR / "games_metadata_processed.csv",
        index=False
    )
    # Salva os jogos processados em CSV.
    #
    # index=False evita salvar o índice do DataFrame como uma coluna.


    processed_ratings.to_csv(
        PROCESSED_DIR / "ratings_processed.csv",
        index=False
    )
    # Salva as avaliações processadas.


    return games, ratings, processed_games, processed_ratings, user_threshold, game_threshold
    # Retorna:
    #
    # dados originais
    # dados processados
    # limite de usuários
    # limite de jogos


def print_stats(label, ratings):
    # Exibe estatísticas formatadas no terminal.


    stats = dataset_stats(ratings)
    # Calcula as estatísticas.


    print(f"\n{label}")
    # Imprime o nome do dataset.
    #
    # \n adiciona uma linha em branco antes.


    print(f"usuarios={stats['users']} jogos={stats['games']} ratings={stats['ratings']}")
    # Imprime quantidade de usuários, jogos e avaliações.


    print(f"densidade={stats['density']:.6f} esparsidade={stats['sparsity']:.6f}")
    # Imprime densidade e esparsidade.
    #
    # :.6f significa:
    # número decimal com 6 casas.


    print(f"media ratings/usuario={stats['mean_ratings_user']:.2f} "
          f"media ratings/jogo={stats['mean_ratings_game']:.2f}")
    # Imprime médias com duas casas decimais.


    print(f"overlap 0={stats['overlap'].get(0, 0)} "
          f"1={stats['overlap'].get(1, 0)} 2={stats['overlap'].get(2, 0)} "
          f"3+={sum(value for key, value in stats['overlap'].items() if key >= 3)}")
    # Mostra quantos pares de usuários possuem:
    #
    # 0 jogos em comum
    # 1 jogo em comum
    # 2 jogos em comum
    # 3 ou mais jogos em comum.


if __name__ == "__main__":
    # Esse bloco só é executado quando este arquivo é executado diretamente.
    #
    # Se outro arquivo fizer:
    # import create_processed_dataset
    #
    # esse código não será executado automaticamente.


    raw_games, raw_ratings, processed_games, processed_ratings, user_threshold, game_threshold = create_processed_dataset()
    # Executa todo o processo de criação do dataset processado.


    print(f"Criterio reproduzivel: usuarios >= quantil {USER_PERCENTILE:.0%} ({user_threshold:.0f}) "
          f"e jogos >= quantil {GAME_PERCENTILE:.0%} ({game_threshold:.0f})")
    # Mostra exatamente qual foi o critério utilizado.
    #
    # :.0% transforma:
    # 0.80 -> 80%
    #
    # Isso ajuda a deixar o processamento reproduzível.


    print("nao houve amostragem aleatoria nem ratings inventados")
    # Registra explicitamente que:
    #
    # não houve seleção aleatória;
    # nenhuma avaliação foi criada.


    print_stats("DATASET ORIGINAL (data/raw)", raw_ratings)
    # Mostra as estatísticas do dataset original.


    print_stats("DATASET PROCESSADO (data/processed)", processed_ratings)
    # Mostra as estatísticas do dataset processado.


    print("Arquivos gerados: data/processed/games_metadata_processed.csv, "
          "data/processed/ratings_processed.csv")
    # Informa quais arquivos foram criados.