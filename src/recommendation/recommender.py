import pandas as pd
# Recebe usuário-alvo, vizinhos encontrados por KNN e uma matriz usuário x jogo, candidatos são jogos dos vizinhos que o usuário-alvo não avaliou
def get_candidate_games(
    target_user,
    similar_users,
    user_game_matrix
):
    # Seleciona a linha correspondente ao usuário-alvo.
    target_games = set(
        user_game_matrix.loc[target_user]
        # Remove os jogos que possuem NaN, sobrando apenas os jogos avaliados pelo usuário
        .dropna()  
        # Pega os IDs dos jogos restantes.
        .index
    )
    # Cria um conjunto vazio para os candidatos
    candidate_games = set()
    # Percorre cada vizinho, similaridade e quantidade de jogos em comum
    for user, similarity, common_games in similar_users:
        user_games = set(
            # Pega as avaliações daquele vizinho
            user_game_matrix.loc[user]
            # Mantém somente jogos que ele avaliou
            .dropna()
            .index
        )

        # Adiciona todos os jogos do vizinho aos candidatos
        candidate_games.update(user_games)
    # Remove dos candidatos todos os jogos que o usuário-alvo já avaliou
    candidate_games -= target_games
    # Retorna os jogos que podem ser recomendados.
    return candidate_games

# Calcula a pontuação prevista de cada jogo candidato, maior similaridade, maior peso da avaliação
def calculate_game_scores(
    similar_users,
    candidate_games,
    user_game_matrix
):
    # Dicionário que armazenará game_id com o score previsto
    game_scores = {}
    # Analisa cada jogo candidato.
    for game in candidate_games:
        # Soma ponderada das avaliações
        weighted_sum = 0
        # Soma das similaridades dos usuários que avaliaram o jogo
        similarity_sum = 0

        # Percorre todos os vizinhos
        for user, similarity, common_games in similar_users:      
            # Obtém a avaliação daquele vizinho para o jogo
            rating = user_game_matrix.loc[user, game]
            # Verifica se o vizinho realmente avaliou o jogo
            if pd.notna(rating):
                # Multiplica a avaliação pela similaridade, o que vira contribuição
                weighted_sum += rating * similarity
                # Soma a similaridade utilizada como peso.
                similarity_sum += similarity
                

        # Só calcula o score se pelo menos um vizinho tiver contribuído.
        if similarity_sum > 0:
            # Calcula a média ponderada
            game_scores[game] = weighted_sum / similarity_sum

    # Retorna todos os jogos que conseguiram receber uma pontuação
    return game_scores

 # Recebe um dicionário de jogos e suas pontuações e retorna os melhores N do maior para o menor   
def rank_games(game_scores, n=10):
    ranked_games = sorted(
        # Converte o dicionário em pares (game_id, score)
        game_scores.items(),
        # Define que a ordenação deve usar o segundo elemento score
        key=lambda x: x[1],
        # Ordena de forma decrescente
        reverse=True
    )
    # Retorna somente os primeiros N jogos
    return ranked_games[:n]    

# Calcula recomendações utilizando similaridade entre jogos, o usuario contribui com seus jogos conhecidos para pontuar jogos similares, procuramos jogos parecidos com os jogos que ele já avaliou
def score_item_based_candidates(user_id, ratings, min_similarity=0.0):
    from src.recommendation.similarity import build_item_similarity
    # Cria novamente uma matriz usuário × jogo
    matrix = ratings.pivot(index="user_id", columns="game_id", values="rating")
    # Verifica se o usuário existe na matriz.
    if user_id not in matrix.index:
        # Se não existir, não há como gerar recomendações
        return {}    
    # Calcula a matriz de similaridade jogo × jogo
    similarities = build_item_similarity(ratings)
    # Obtém somente os jogos que o usuário já avaliou
    known = matrix.loc[user_id].dropna()
    # Guarda a soma ponderada das pontuações
    scores = {}
    # Guarda a soma dos pesos/similaridades
    weights = {}
    
    # Percorre cada jogo conhecido pelo usuário
    for known_game, rating in known.items():
        # Percorre todos os jogos e a similaridade deles com o jogo conhecido.
        for candidate, similarity in similarities.loc[known_game].items():

            # Ignora jogos que o usuário já conhece, similaridade baixas e o próprio jogo
            if candidate in known.index or similarity <= min_similarity or candidate == known_game:    
                # Pula para o próximo candidato
                continue   
            # Acumula a contribuição daquele jogo conhecido
            scores[candidate] = scores.get(candidate, 0.0) + similarity * rating
            # Acumula a similaridade usada como peso
            weights[candidate] = weights.get(candidate, 0.0) + similarity
            

    # Calcula a média ponderada para cada candidato que só mantêm o candidato seja maior que zero
    return {game: scores[game] / weights[game] for game in scores if weights[game] > 0}

# Função para gerar recomendações item-based
def recommend_item_based(user_id, ratings, n=10, min_similarity=0.0):
    # Primeiro calcula scores e depois ordena e pega os N melhores
    return rank_games(score_item_based_candidates(user_id, ratings, min_similarity), n)

# Gera recomendações para um único usuário, mínimo de jogos para considerar e quant máxima de vizinhos
def recommend_for_user(user_id, ratings, games=None, k=10, min_common_games=2, neighbor_count=10):
    from src.data.preprocess import create_user_game_matrix
    from src.recommendation.similarity import find_similar_users

    matrix = create_user_game_matrix(ratings)
    # Verifica se o usuário existe na matriz
    if user_id not in matrix.index:
        # Se não existir, não existem dados para recomendar
        return []
        
    # Encontra os usuários mais similares ao usuário-alvo
    neighbors = find_similar_users(
        user_id,
        matrix,
        k=neighbor_count,
        min_common_games=min_common_games
    )
    # Pega jogos dos vizinhos que o usuário ainda não avaliou
    candidates = get_candidate_games(user_id, neighbors, matrix)
    
    # Calcula uma pontuação prevista para cada candidato
    scores = calculate_game_scores(neighbors, candidates, matrix)
    
    # Ordena os candidatos e seleciona os K melhores
    ranked = rank_games(scores, k)
    
    # Verifica se metadata dos jogos foi fornecida
    if games is None:
        # Se não houver, retorna somente game_id e score
        return [{"game_id": int(game), "score": float(score)} for game, score in ranked]

    # Transforma a tabela de jogos em um dicionário indexado por game_id
    metadata = games.set_index("game_id").to_dict("index")
    # Para cada recomendação pega game_id, informações do jogo e score
    return [{"game_id": int(game), **metadata.get(game, {}), "score": float(score)} for game, score in ranked]