import pandas as pd
from src.data.preprocess import create_user_game_matrix
from src.recommendation.recommender import calculate_game_scores, get_candidate_games, rank_games
from src.recommendation.similarity import find_similar_users

# Divide as avaliações de um usuário em treinamento e teste, 20% pra teste, random_state irá esccolher sempre o mesmo conjunto
def split_user_ratings(user_id, ratings, test_size=0.2, random_state=42):
    # Seleciona somente as avaliações do usuário escolhido.
    user_ratings = ratings[ratings["user_id"] == user_id].copy()
    # Verifica se o usuário possui pelo menos duas avaliações.
    if len(user_ratings) < 2:
        # Sem pelo menos duas avaliações não é possível separar adequedamente treinamento e teste
        raise ValueError("O usuario precisa de pelo menos duas avaliacoes")
        
    # Calcula quantas avaliações irão para o teste, max(1) garante pelo menos uma avaliação no teste, test_size = 20%
    test_count = max(1, int(round(len(user_ratings) * test_size)))
    
    # Seleciona aleatoriamente test_count avaliações com o mesmo conjunto 
    test_ratings = user_ratings.sample(n=test_count, random_state=random_state)
    # Retorna avaliações de treinamento primeiro e depois teste
    return user_ratings.drop(test_ratings.index), test_ratings

# Calcula Precision@K, dos K jogos recomendados, quantos são relevantes?
def precision_at_k(recommendations, relevant_games, k):
    # Verifica se K é zero ou negativo.
    if k <= 0:
        # Evita uma divisão inválida.
        return 0.0   
    # Pega somente os primeiros K jogos recomendados
    recommended = {game for game, _ in recommendations[:k]}
    
    # Calcula a interseção entre jogos recomendados e relevantes e depois divide por K
    return len(recommended & set(relevant_games)) / k

# Calcula Recall@K mede a proporcao dos relevantes recuperados no Top-K, dos jogos relevantes que existiam, quantos o sistema conseguiu encontrar?
def recall_at_k(recommendations, relevant_games, k):
    # Transforma os jogos relevantes em um conjunto.
    relevant = set(relevant_games)
    if not relevant:
        # Não existe o que recuperar
        return 0.0
    # Pega somente os K jogos recomendados
    recommended = {game for game, _ in recommendations[:k]}
    # Calcula quantos relevantes foram encontrados dividido pela quantidade total de relevantes
    return len(recommended & relevant) / len(relevant)

# Cria recomendações baseadas em popularidade somente a partir das avaliacoes de treino para evitar que as avaliações do conjunto de teste influenciem a recomendação
def popular_recommendations(user_id, train_ratings, k=10, min_count=1):
    # Encontra os jogos que o usuário já conhece/avaliou e transforma em conjunto
    known = set(train_ratings.loc[train_ratings["user_id"] == user_id, "game_id"])
    # Agrupa as avaliações por jogo, mean = média de avaliação e count = quant. de avaliação
    stats = train_ratings.groupby("game_id")["rating"].agg(["mean", "count"])
    # Remove jogos que possuem menos avaliações que o mínimo definido
    stats = stats[stats["count"] >= min_count].copy()
    # Cria uma pontuação de popularidade para desempate de jogos avaliados
    stats["score"] = stats["mean"] + 0.01 * stats["count"]
    # Remove os jogos que o usuário já conhece, depois ordena
    stats = stats.drop(index=known, errors="ignore").sort_values("score", ascending=False)

    # Pega os primeiros K jogos e retorna uma lista com (game_id, score),...
    return [(game, float(row.score)) for game, row in stats.head(k).iterrows()]

# Avalia um modelo de recomendação para um usuário
def evaluate_user(user_id, ratings, k=10, min_common_games=2, neighbor_count=10,
                  test_size=0.2, random_state=42, model="user"):
    
    # Divide as avaliações do usuário em treino e teste.
    train_user, test_ratings = split_user_ratings(user_id, ratings, test_size, random_state)
    # Remove do conjunto completo as avaliações que foram colocadas no teste
    ratings_train = ratings.drop(test_ratings.index)
    # Cria a matriz usuário × jogo usando somente o treinamento
    matrix = create_user_game_matrix(ratings_train)
    # Define como relevantes os jogos que o usuário avaliou com 4 ou 5 nos testes
    relevant = set(test_ratings.loc[test_ratings["rating"] >= 4, "game_id"])
    # Inicializa a lista de vizinhos para o modelo de filtragem colaborativa
    neighbors = []
    # Inicializa o conjunto de jogos candidatos
    candidates = set()
    # Inicializa um dicionário que armazenará quantos vizinhos contribuíram para cada recomendação
    contributors = {}

    # Entra aqui quando o modelo escolhido é para filtragem colaborativa e o usuário está presente na matriz
    if model == "user" and user_id in matrix.index:
        # Encontra os usuários mais similares ao usuário-alvo
        neighbors = find_similar_users(user_id, matrix, neighbor_count, min_common_games)
        
        # Encontra jogos avaliados pelos vizinhos que o usuário-alvo ainda não avaliou
        candidates = get_candidate_games(user_id, neighbors, matrix)

        # Calcula a pontuação prevista para cada jogo candidato
        scores = calculate_game_scores(neighbors, candidates, matrix)

        # Ordena as pontuações e pega os primeiros jogos K
        recommendations = rank_games(scores, k)
        
        # Conta quantos vizinhos possuem uma avaliação para aquele jogo
        contributors = {game: sum(pd.notna(matrix.loc[user, game]) for user, _, _ in neighbors)
                        for game in candidates}

    elif model == "popularity":
        # Gera recomendações usando somente popularidade.
        recommendations = popular_recommendations(user_id, ratings_train, k)
        
    # Caso o modelo escolhido seja item-based
    elif model == "item":
        # Importa a função responsável por calcular scores baseados na similaridade entre jogos
        from src.recommendation.recommender import score_item_based_candidates
        # Calcula a pontuação dos jogos candidatos
        scores = score_item_based_candidates(user_id, ratings_train)
        # Pega os IDs dos jogos que receberam pontuação
        candidates = set(scores)
        recommendations = rank_games(scores, k)
    else:
        raise ValueError("model invalido")
    
    # Extrai somente os IDs dos jogos recomendados
    recommended_ids = {game for game, _ in recommendations}
    
    # Retorna um dicionário contendo diversas métricas da avaliação
    return {
        # Identifica qual usuário foi avaliado
        "user_id": user_id,
        # Quantidade total de avaliações que o usuário possui   
        "total_ratings": len(ratings[ratings["user_id"] == user_id]),
        # Quantidade de avaliações utilizadas no treinamento
        "train_ratings": len(train_user),
        # Quantidade de avaliações reservadas para teste
        "test_ratings": len(test_ratings),
        # Quantidade de jogos considerados relevantes
        "relevant_count": len(relevant),
        # Quantidade de vizinhos encontrados
        "neighbors": len(neighbors),
        # Quantidade de jogos candidatos
        "candidate_count": len(candidates),
        # Quantidade de recomendações geradas
        "recommendation_count": len(recommendations),
        # Jogos que aparecem tanto nas recomendações quanto nos jogos relevantes
        "intersection": sorted(recommended_ids & relevant),
        "precision@5": precision_at_k(recommendations, relevant, 5),
        "precision@10": precision_at_k(recommendations, relevant, 10),
        "recall@5": recall_at_k(recommendations, relevant, 5),
        "recall@10": recall_at_k(recommendations, relevant, 10),
        "coverage": bool(recommendations),
        # Calcula a média de contribuidores por jogo.
        "mean_contributors": (sum(contributors.values()) / len(contributors) if contributors else 0.0),
        # Calcula a proporção dos candidatos que receberam contribuição de exatamente 1 vizinho
        "one_contributor_ratio": (sum(value == 1 for value in contributors.values()) / len(contributors)
                                  if contributors else 0.0),
        "two_contributor_ratio": (sum(value == 2 for value in contributors.values()) / len(contributors)
                                  if contributors else 0.0),
        "three_plus_contributor_ratio": (sum(value >= 3 for value in contributors.values()) / len(contributors)
                                         if contributors else 0.0),
        # Guarda a lista completa das recomendações
        "recommendations": recommendations,
        # Guarda os jogos considerados relevantes, ordenados pelos IDs.
        "relevant_games": sorted(relevant),
        # Guarda a quantidade de contribuidores por jogo.
        "contributors": contributors,
    }

# Avalia os três modelos para vários usuários
def evaluate_models(ratings, user_ids, min_common_games=2, neighbor_count=10,
                    test_size=0.2, random_state=42):
    # Lista onde serão armazenados os resultados individuais
    rows = []   
    # Percorre os três modelos
    for model in ("popularity", "user", "item"): 
        # Para cada modelo, percorre todos os usuários escolhidos
        for user_id in user_ids:
            # Avalia aquele usuário usando aquele modelo
            row = evaluate_user(user_id, ratings, 10, min_common_games, neighbor_count,
                                test_size, random_state, model)
            # Adiciona ao resultado qual modelo foi utilizado
            row["model"] = model       
            # Adiciona o resultado à lista
            rows.append(row)
            
    # Transforma todos os resultados individuais em um DataFrame
    details = pd.DataFrame(rows)
    # Agrupa os resultados por modelo e calcula médias
    summary = details.groupby("model").agg(
        # Média da Precision@5.
        precision_at_5=("precision@5", "mean"),
        precision_at_10=("precision@10", "mean"),
        recall_at_5=("recall@5", "mean"),
        recall_at_10=("recall@10", "mean"),
        coverage=("coverage", "mean"),
        # Média da quantidade de vizinhos.
        mean_neighbors=("neighbors", "mean"),
        # Média da quantidade de candidatos.
        mean_candidates=("candidate_count", "mean"),
        # Média da quantidade de recomendações.
        mean_recommendations=("recommendation_count", "mean"),
        # Média dos contribuidores.
        mean_contributors=("mean_contributors", "mean"), 
        # Média da proporção de jogos com um contribuidor
        one_contributor_ratio=("one_contributor_ratio", "mean"),
        two_contributor_ratio=("two_contributor_ratio", "mean"),
        three_plus_contributor_ratio=("three_plus_contributor_ratio", "mean"),
    # reset_index() transforma o índice "model" novamente em uma coluna.
    ).reset_index()
    
    # Retorna resumo por modelo e resultados individuais de cada usuário
    return summary, details
