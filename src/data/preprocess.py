import pandas as pd

# Define uma função que transforma a tabela de avaliações em uma matriz usuário × jogo
def create_user_game_matrix(ratings):

    # pivot reorganiza a tabela
    return ratings.pivot(
        # Define que cada linha da nova matriz representa um usuário
        index="user_id",
        # Define que cada coluna representa um jogo
        columns="game_id",
        # Define que o conteúdo das células será a avaliação
        values="rating"   
    )

# Verifica se os dados possuem a estrutura e os valores esperados.
def validate_data(games, ratings):
    # Tabela games com colunas game_id
    required_games = {"game_id"}
    # Tabela ratings com colunas user_id, game_id e rating
    required_ratings = {"user_id", "game_id", "rating"}
    missing = {
        # Vê se nas colunas possuem o mesmo tipo de informação que está em required("game_id") mas não em columns("name_id") = games_id
        "games": required_games - set(games.columns),   
        "ratings": required_ratings - set(ratings.columns),
    }

    # Checa se há valores ausentes nas colunas
    if any(missing.values()):
        # Interrompe a execução informando quais colunas estão faltando com informação
        raise ValueError(f"Colunas ausentes: {missing}")
        

    # Verifica se existem valores nulos em campos, se há algum em games, ou required com 3 colunas de user_id, game_id e rating
    if games["game_id"].isna().any() or ratings[list(required_ratings)].isna().any().any():
        raise ValueError("IDs e ratings principais nao podem ser nulos")

    # Verifica se o mesmo usuário avaliou o mesmo jogo mais de uma vez
    if ratings.duplicated(["user_id", "game_id"]).any():
        raise ValueError("Existem avaliacoes duplicadas para o mesmo usuario e jogo")
    # Verifica se todos os game_ids existentes em ratings # também existem em games, todos existem -> False
    if not ratings["game_id"].isin(games["game_id"]).all():
        # Informa que existe uma avaliação para um jogo que não possui informações na tabela games
        raise ValueError("Existem ratings sem metadata correspondente") 

    # Verifica se todas as avaliações estão entre 1 e 5
    if not ratings["rating"].between(1, 5).all():
        raise ValueError("Ratings devem estar entre 1 e 5")
     # Se nenhuma validação falhou, retorna True
    return True
   
