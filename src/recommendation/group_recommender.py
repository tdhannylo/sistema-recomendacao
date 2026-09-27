from collections import defaultdict
# Importa defaultdict.
#
# Ele funciona como um dicionário, mas pode criar automaticamente
# um valor padrão quando uma chave ainda não existe.


from src.recommendation.recommender import recommend_for_user
# Importa a função que recomenda jogos individualmente.


def recommend_for_group(user_ids, ratings, games=None, k=10, strategy="mean", **kwargs):
    # Define a função responsável pela recomendação para o grupo.
    #
    # user_ids -> IDs dos 3 usuários.
    # ratings -> avaliações.
    # games -> metadata opcional.
    # k -> quantidade final de recomendações.
    # strategy -> forma de combinar as recomendações.
    # **kwargs -> permite passar parâmetros adicionais para recommend_for_user.


    # Cada membro e recomendado separadamente antes da agregacao do grupo.
    # Primeiro cada usuário recebe suas próprias recomendações.
    # Só depois elas serão combinadas.


    if len(user_ids) != 3:
        # Verifica se existem exatamente 3 usuários.


        raise ValueError("O grupo deve conter exatamente 3 usuarios")
        # Impede grupos com quantidade diferente de 3.


    per_user = [recommend_for_user(user_id, ratings, k=max(k, 50), **kwargs) for user_id in user_ids]
    # Gera recomendações individualmente para cada usuário.
    #
    # max(k, 50) significa que o sistema pede pelo menos 50 recomendações
    # para cada usuário antes de fazer a agregação.
    #
    # Isso fornece uma quantidade maior de candidatos para o grupo.
    #
    # O resultado será uma lista contendo as recomendações de cada usuário.


    # Remove jogos que qualquer membro do grupo ja conhece.
    # O grupo não deve receber um jogo que algum dos participantes
    # já avaliou.


    known = set(ratings[ratings["user_id"].isin(user_ids)]["game_id"])
    # Seleciona todas as avaliações dos usuários do grupo.
    #
    # .isin(user_ids) verifica se o user_id pertence ao grupo.
    #
    # Depois pega somente game_id.
    #
    # set() transforma tudo em um conjunto de jogos conhecidos.


    scores = defaultdict(list)
    # Cria um dicionário em que cada jogo terá uma lista de scores.
    #
    # Exemplo:
    #
    # scores[123] = [4.8, 4.5, 4.9]


    details = {}
    # Dicionário para armazenar os detalhes de cada recomendação.


    for recommendations in per_user:
        # Percorre as recomendações de cada usuário.


        for recommendation in recommendations:
            # Percorre cada recomendação individual.


            game = recommendation["game_id"]
            # Obtém o ID do jogo.


            if game not in known:
                # Só continua se nenhum membro do grupo já conhecer
                # aquele jogo.


                scores[game].append(recommendation["score"])
                # Adiciona o score daquele usuário à lista do jogo.


                details[game] = recommendation
                # Guarda os detalhes da recomendação.


    if strategy == "mean":
        # Verifica se a estratégia escolhida é média.


        # A media busca uma boa pontuacao conjunta para os tres usuarios.
        # O score final será a média dos scores disponíveis.


        aggregate = {game: sum(values) / len(values) for game, values in scores.items()}
        # Para cada jogo:
        #
        # score final = soma dos scores / quantidade de scores.
        #
        # Exemplo:
        # usuário 1 -> 5
        # usuário 2 -> 4
        # usuário 3 -> 3
        #
        # média = 4.


    elif strategy == "minimum":
        # Verifica se a estratégia escolhida é minimum.


        aggregate = {game: min(values) for game, values in scores.items()}
        # Usa o menor score disponível para cada jogo.
        #
        # Exemplo:
        # [5, 4, 3] -> 3
        #
        # Isso busca evitar que um jogo tenha score muito baixo
        # para algum participante.


    else:
        # Caso strategy não seja "mean" nem "minimum".


        raise ValueError("strategy deve ser 'mean' ou 'minimum'")
        # Informa que a estratégia é inválida.


    ranked = sorted(aggregate.items(), key=lambda item: item[1], reverse=True)[:k]
    # Ordena os jogos pelo score final.
    #
    # item[0] -> game_id
    # item[1] -> score
    #
    # reverse=True -> maior para menor.
    #
    # [:k] -> pega apenas os primeiros K.


    result = []
    # Cria a lista final de recomendações.


    for game, score in ranked:
        # Percorre os jogos ordenados.


        item = dict(details[game])
        # Cria uma cópia dos detalhes daquele jogo.


        item["score"] = float(score)
        # Atualiza o score para o score agregado do grupo.


        item["group_score"] = float(score)
        # Também salva explicitamente o score como group_score.


        result.append(item)
        # Adiciona o resultado à lista final.


    return result
    # Retorna as recomendações finais do grupo.