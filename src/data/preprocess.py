import pandas as pd
# Importa pandas.
#
# Neste trecho específico, o pandas não é usado diretamente
# dentro de create_user_game_matrix(), mas pode ser necessário
# em outras partes desse módulo.


def create_user_game_matrix(ratings):
    # Define uma função que transforma a tabela de avaliações
    # em uma matriz usuário × jogo.


    # Cada linha representa um usuario e cada coluna representa um jogo.
    # Explica a estrutura que será criada.


    return ratings.pivot(
        # pivot reorganiza a tabela para transformar:
        #
        # linhas originais:
        # user_id | game_id | rating
        #
        # em:
        #           jogo1  jogo2  jogo3
        # usuario1    5      3      NaN
        # usuario2    4      NaN    5


        index="user_id",
        # Define que cada linha da nova matriz representa um usuário.


        columns="game_id",
        # Define que cada coluna representa um jogo.


        values="rating"
        # Define que o conteúdo das células será a avaliação.
    )

def validate_data(games, ratings):
    # Cria uma função para verificar se os dados possuem
    # a estrutura e os valores esperados.


    required_games = {"game_id"}
    # Define as colunas obrigatórias da tabela games.
    #
    # Para games, precisamos pelo menos do game_id.


    required_ratings = {"user_id", "game_id", "rating"}
    # Define as colunas obrigatórias da tabela ratings.


    missing = {
        "games": required_games - set(games.columns),
        # games.columns contém as colunas existentes.
        # set(games.columns) transforma essas colunas em um conjunto.
        #
        # A operação "-" encontra aquilo que está em required_games
        # mas não existe em games.
        #
        # Exemplo:
        # required = {"game_id"}
        # existente = {"name", "rating"}
        #
        # resultado:
        # {"game_id"}


        "ratings": required_ratings - set(ratings.columns),
        # Faz a mesma verificação para ratings.
    }


    if any(missing.values()):
        # missing.values() contém os conjuntos de colunas ausentes.
        #
        # any() verifica se pelo menos um deles contém algo.
        #
        # Se alguma coluna obrigatória estiver faltando, entra no if.


        raise ValueError(f"Colunas ausentes: {missing}")
        # Interrompe a execução informando quais colunas estão faltando.


    if games["game_id"].isna().any() or ratings[list(required_ratings)].isna().any().any():
        # Verifica se existem valores nulos em campos importantes.
        #
        # games["game_id"].isna()
        # -> identifica game_ids nulos.
        #
        # .any()
        # -> verifica se existe pelo menos um.
        #
        # ratings[list(required_ratings)]
        # -> seleciona user_id, game_id e rating.
        #
        # .isna().any().any()
        # -> verifica se existe algum valor nulo nessas colunas.


        raise ValueError("IDs e ratings principais nao podem ser nulos")
        # Interrompe a execução se houver valores nulos.


    if ratings.duplicated(["user_id", "game_id"]).any():
        # Verifica se o mesmo usuário avaliou o mesmo jogo mais de uma vez.
        #
        # ["user_id", "game_id"] define a combinação que deveria ser única.


        raise ValueError("Existem avaliacoes duplicadas para o mesmo usuario e jogo")
        # Gera erro caso existam avaliações duplicadas.


    if not ratings["game_id"].isin(games["game_id"]).all():
        # Verifica se TODOS os game_ids existentes em ratings
        # também existem em games.
        #
        # .isin(...) retorna True/False para cada rating.
        #
        # .all() verifica se todos são True.
        #
        # not transforma:
        # todos existem -> False
        # algum não existe -> True


        raise ValueError("Existem ratings sem metadata correspondente")
        # Informa que existe uma avaliação para um jogo que
        # não possui informações na tabela games.


    if not ratings["rating"].between(1, 5).all():
        # Verifica se todas as avaliações estão entre 1 e 5.


        raise ValueError("Ratings devem estar entre 1 e 5")
        # Gera erro se existir uma avaliação fora desse intervalo.


    return True
    # Se nenhuma validação falhou, retorna True.
