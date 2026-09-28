from collections import defaultdict
from src.recommendation.recommender import recommend_for_user

# Define a função responsável pela recomendação para o grupo
def recommend_for_group(user_ids, ratings, games=None, k=10, strategy="mean", **kwargs):
    # Verifica se existem exatamente 3 usuários
    if len(user_ids) != 3:
        raise ValueError("O grupo deve conter exatamente 3 usuarios")
    # Gera recomendações individualmente para cada usuário, uma lista de 50 para cada usuário
    per_user = [recommend_for_user(user_id, ratings, k=max(k, 50), **kwargs) for user_id in user_ids]
    # Seleciona todas as avaliações dos usuários do grupo, caso tiver, pega o game_id
    known = set(ratings[ratings["user_id"].isin(user_ids)]["game_id"])
    # Cria um dicionário em que cada jogo terá uma lista de scores de recomendação
    scores = defaultdict(list)
    # Dicionário para armazenar os detalhes de cada recomendação.
    details = {}

    # Percorre as recomendações de cada usuário.
    for recommendations in per_user:
        # Percorre cada recomendação individual.
        for recommendation in recommendations:
            # Obtém o ID do jogo
            game = recommendation["game_id"]
            # Só continua se nenhum membro do grupo já conhecer aquele jogo
            if game not in known:
                # Adiciona o score daquele usuário à lista do jogo
                scores[game].append(recommendation["score"])
                # Guarda os detalhes da recomendação
                details[game] = recommendation
                
    # Verifica se a estratégia escolhida é média com uma boa pontuação para os 3 usuários
    if strategy == "mean":
        # Para cada jogo score final = soma dos scores / quantidade de scores.
        aggregate = {game: sum(values) / len(values) for game, values in scores.items()}

    # Verifica se a estratégia escolhida é minimum
    elif strategy == "minimum":
        # Usa o menor score disponível para cada jogo para evitar que um jogo tenha um score baixo para um usuário
        aggregate = {game: min(values) for game, values in scores.items()}
    else:
        raise ValueError("strategy deve ser 'mean' ou 'minimum'")
    
    # Ordena os jogos pelo score final, :k pega os primeiros K do maior para o menor
    ranked = sorted(aggregate.items(), key=lambda item: item[1], reverse=True)[:k]
    # Cria a lista final de recomendações.
    result = []
    
    # Percorre os jogos ordenados.
    for game, score in ranked:  
        # Cria uma cópia dos detalhes daquele jogo
        item = dict(details[game])
        # Atualiza o score para o score agregado do grupo
        item["score"] = float(score)
        # Também salva explicitamente o score como group_score
        item["group_score"] = float(score)
        # Adiciona o resultado à lista final.
        result.append(item)
    return result