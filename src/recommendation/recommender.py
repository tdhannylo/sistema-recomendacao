import pandas as pd
# Importa pandas.
# É utilizado posteriormente para verificar valores NaN.


def get_candidate_games(
    target_user,
    similar_users,
    user_game_matrix
):
    # Recebe:
    #
    # target_user -> usuário para quem queremos recomendar.
    # similar_users -> vizinhos encontrados pelo KNN.
    # user_game_matrix -> matriz usuário × jogo.


    # Candidatos sao jogos dos vizinhos que o usuario-alvo ainda nao avaliou.
    # Essa é a ideia principal da geração de candidatos.


    target_games = set(
        user_game_matrix.loc[target_user]
        # Seleciona a linha correspondente ao usuário-alvo.


        .dropna()
        # Remove os jogos que possuem NaN.
        # Sobram apenas jogos que o usuário avaliou.


        .index
        # Pega os IDs dos jogos restantes.
    )


    candidate_games = set()
    # Cria um conjunto vazio para os candidatos.


    for user, similarity, common_games in similar_users:
        # Percorre cada vizinho.
        #
        # user -> ID do vizinho.
        # similarity -> similaridade com o alvo.
        # common_games -> quantidade de jogos em comum.


        user_games = set(
            user_game_matrix.loc[user]
            # Pega as avaliações daquele vizinho.


            .dropna()
            # Mantém somente jogos que ele avaliou.


            .index
            # Obtém os IDs desses jogos.
        )


        candidate_games.update(user_games)
        # Adiciona todos os jogos do vizinho aos candidatos.
        #
        # update() adiciona vários elementos ao set.


    candidate_games -= target_games
    # Remove dos candidatos todos os jogos que o usuário-alvo
    # já avaliou.


    return candidate_games
    # Retorna os jogos que podem ser recomendados.

def calculate_game_scores(
    similar_users,
    candidate_games,
    user_game_matrix
):
    # Calcula a pontuação prevista de cada jogo candidato.


    # A nota prevista e a media ponderada pelas similaridades dos vizinhos.
    # Quanto maior a similaridade do vizinho,
    # maior o peso da avaliação dele.


    game_scores = {}
    # Dicionário que armazenará:
    #
    # game_id -> score previsto


    for game in candidate_games:
        # Analisa cada jogo candidato.


        weighted_sum = 0
        # Soma ponderada das avaliações.


        similarity_sum = 0
        # Soma das similaridades dos usuários que avaliaram o jogo.


        for user, similarity, common_games in similar_users:
            # Percorre todos os vizinhos.


            rating = user_game_matrix.loc[user, game]
            # Obtém a avaliação daquele vizinho para o jogo.


            if pd.notna(rating):
                # Verifica se o vizinho realmente avaliou o jogo.


                weighted_sum += rating * similarity
                # Multiplica a avaliação pela similaridade.
                #
                # Exemplo:
                # rating = 5
                # similarity = 0.8
                #
                # contribuição = 4.0


                similarity_sum += similarity
                # Soma a similaridade utilizada como peso.


        if similarity_sum > 0:
            # Só calcula o score se pelo menos um vizinho
            # tiver contribuído.


            game_scores[game] = weighted_sum / similarity_sum
            # Calcula a média ponderada.
            #
            # Fórmula:
            #
            # score =
            # Σ(rating × similarity)
            # ----------------------
            # Σ similarity


    return game_scores
    # Retorna todos os jogos que conseguiram receber uma pontuação.

def rank_games(game_scores, n=10):
    # Recebe um dicionário de jogos e suas pontuações
    # e retorna os melhores N.


    # Ordena as previsoes da maior para a menor nota.
    # Portanto, o primeiro jogo será o de maior score.


    ranked_games = sorted(
        game_scores.items(),
        # Converte o dicionário em pares:
        #
        # (game_id, score)


        key=lambda x: x[1],
        # Define que a ordenação deve usar o segundo elemento:
        # o score.


        reverse=True
        # Ordena de forma decrescente.
    )


    return ranked_games[:n]
    # Retorna somente os primeiros N jogos.


def score_item_based_candidates(user_id, ratings, min_similarity=0.0):
    # Calcula recomendações utilizando similaridade entre jogos.


    from src.recommendation.similarity import build_item_similarity
    # Importa a função que calcula a matriz de similaridade
    # entre jogos.


    # O usuario contribui com seus jogos conhecidos para pontuar jogos similares.
    # Ou seja, em vez de procurar usuários parecidos,
    # procuramos jogos parecidos com os jogos que ele já avaliou.


    matrix = ratings.pivot(index="user_id", columns="game_id", values="rating")
    # Cria novamente uma matriz usuário × jogo.


    if user_id not in matrix.index:
        # Verifica se o usuário existe na matriz.


        return {}
        # Se não existir, não há como gerar recomendações.


    similarities = build_item_similarity(ratings)
    # Calcula a matriz de similaridade jogo × jogo.


    known = matrix.loc[user_id].dropna()
    # Obtém somente os jogos que o usuário já avaliou.


    scores = {}
    # Guarda a soma ponderada das pontuações.


    weights = {}
    # Guarda a soma dos pesos/similaridades.


    for known_game, rating in known.items():
        # Percorre cada jogo conhecido pelo usuário.


        for candidate, similarity in similarities.loc[known_game].items():
            # Percorre todos os jogos e a similaridade deles
            # com o jogo conhecido.


            if candidate in known.index or similarity <= min_similarity or candidate == known_game:
                # Ignora:
                #
                # jogos que o usuário já conhece;
                # similaridades muito baixas;
                # o próprio jogo.


                continue
                # Pula para o próximo candidato.


            scores[candidate] = scores.get(candidate, 0.0) + similarity * rating
            # Acumula a contribuição daquele jogo conhecido.
            #
            # similaridade × rating


            weights[candidate] = weights.get(candidate, 0.0) + similarity
            # Acumula a similaridade usada como peso.


    return {game: scores[game] / weights[game] for game in scores if weights[game] > 0}
    # Calcula a média ponderada para cada candidato.
    #
    # Só mantém candidatos cujo peso total seja maior que zero.


def recommend_item_based(user_id, ratings, n=10, min_similarity=0.0):
    # Função de conveniência para gerar recomendações item-based.


    return rank_games(score_item_based_candidates(user_id, ratings, min_similarity), n)
    # Primeiro:
    #
    # score_item_based_candidates(...)
    # -> calcula os scores.
    #
    # Depois:
    #
    # rank_games(...)
    # -> ordena e pega os N melhores.


def recommend_for_user(user_id, ratings, games=None, k=10, min_common_games=2, neighbor_count=10):
    # Gera recomendações para um único usuário.
    #
    # user_id -> usuário-alvo.
    # ratings -> avaliações.
    # games -> metadata opcional.
    # k -> quantidade de recomendações.
    # min_common_games -> mínimo de jogos em comum para considerar
    #                     outro usuário como vizinho.
    # neighbor_count -> quantidade máxima de vizinhos.


    from src.data.preprocess import create_user_game_matrix
    # Importa a função que cria a matriz usuário × jogo.


    from src.recommendation.similarity import find_similar_users
    # Importa a função que encontra usuários similares.


    # Fluxo user-based: matriz, vizinhos, candidatos, scores e ranking final.
    # Essa linha resume todo o pipeline.


    matrix = create_user_game_matrix(ratings)
    # Cria a matriz usuário × jogo.


    if user_id not in matrix.index:
        # Verifica se o usuário existe na matriz.


        return []
        # Se não existir, não existem dados para recomendar.


    neighbors = find_similar_users(
        user_id,
        matrix,
        k=neighbor_count,
        min_common_games=min_common_games
    )
    # Encontra os usuários mais similares ao usuário-alvo.


    candidates = get_candidate_games(user_id, neighbors, matrix)
    # Pega jogos dos vizinhos que o usuário ainda não avaliou.


    scores = calculate_game_scores(neighbors, candidates, matrix)
    # Calcula uma pontuação prevista para cada candidato.


    ranked = rank_games(scores, k)
    # Ordena os candidatos e seleciona os K melhores.


    if games is None:
        # Verifica se metadata dos jogos foi fornecida.


        return [{"game_id": int(game), "score": float(score)} for game, score in ranked]
        # Se não houver metadata, retorna somente:
        #
        # game_id
        # score


    metadata = games.set_index("game_id").to_dict("index")
    # Transforma a tabela de jogos em um dicionário indexado por game_id.
    #
    # Exemplo:
    #
    # {
    #   123: {"name": "Jogo A", "released": "..."},
    #   456: {"name": "Jogo B", "released": "..."}
    # }


    return [{"game_id": int(game), **metadata.get(game, {}), "score": float(score)} for game, score in ranked]
    # Para cada recomendação:
    #
    # pega game_id
    # +
    # informações do jogo
    # +
    # score
    #
    # **metadata.get(game, {})
    # "espalha" as informações do dicionário dentro do resultado.