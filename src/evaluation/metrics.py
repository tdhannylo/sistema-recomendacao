import pandas as pd

from src.data.preprocess import create_user_game_matrix
from src.recommendation.recommender import calculate_game_scores, get_candidate_games, rank_games
from src.recommendation.similarity import find_similar_users


def split_user_ratings(user_id, ratings, test_size=0.2, random_state=42):
    # Divide as avaliações de um usuário em:
    # treinamento e teste.
    #
    # test_size=0.2 significa aproximadamente 20% para teste.
    #
    # random_state=42 garante que a divisão seja reproduzível.


    # O split e feito por usuario para avaliar jogos removidos do treino.
    # A ideia é retirar algumas avaliações do usuário
    # e verificar depois se o sistema consegue recomendar esses jogos.


    user_ratings = ratings[ratings["user_id"] == user_id].copy()
    # Seleciona somente as avaliações do usuário escolhido.
    #
    # .copy() cria uma cópia independente do DataFrame.


    if len(user_ratings) < 2:
        # Verifica se o usuário possui pelo menos duas avaliações.


        raise ValueError("O usuario precisa de pelo menos duas avaliacoes")
        # Sem pelo menos duas avaliações não é possível separar
        # adequadamente treinamento e teste.


    test_count = max(1, int(round(len(user_ratings) * test_size)))
    # Calcula quantas avaliações irão para o teste.
    #
    # len(user_ratings) -> quantidade total de avaliações.
    #
    # * test_size -> aproximadamente 20%.
    #
    # round() -> arredonda.
    #
    # int() -> transforma em inteiro.
    #
    # max(1, ...) -> garante pelo menos uma avaliação no teste.


    test_ratings = user_ratings.sample(n=test_count, random_state=random_state)
    # Seleciona aleatoriamente test_count avaliações.
    #
    # random_state garante que o mesmo conjunto seja escolhido
    # sempre que o código for executado com os mesmos dados.


    return user_ratings.drop(test_ratings.index), test_ratings
    # Retorna dois DataFrames:
    #
    # primeiro -> avaliações de treinamento
    # segundo -> avaliações de teste.


def precision_at_k(recommendations, relevant_games, k):
    # Calcula Precision@K.


    # Mede a proporcao do Top-K que pertence ao conjunto relevante.
    # Ou seja:
    #
    # dos K jogos recomendados,
    # quantos realmente pertenciam ao conjunto considerado relevante?


    if k <= 0:
        # Verifica se K é zero ou negativo.


        return 0.0
        # Evita uma divisão inválida.


    recommended = {game for game, _ in recommendations[:k]}
    # Pega somente os primeiros K jogos recomendados.
    #
    # recommendations possui algo como:
    #
    # [(123, 4.9), (456, 4.8), (789, 4.7)]
    #
    # O _ representa a pontuação, que não será usada aqui.
    #
    # Resultado:
    # {123, 456, 789}


    return len(recommended & set(relevant_games)) / k
    # & calcula a interseção entre:
    #
    # jogos recomendados
    # e
    # jogos relevantes.
    #
    # Depois divide pela quantidade K.


def recall_at_k(recommendations, relevant_games, k):
    # Calcula Recall@K.


    # Mede a proporcao dos relevantes recuperados no Top-K.
    # Diferentemente de precision, aqui a pergunta é:
    #
    # "Dos jogos relevantes que existiam,
    # quantos o sistema conseguiu encontrar?"


    relevant = set(relevant_games)
    # Transforma os jogos relevantes em um conjunto.


    if not relevant:
        # Verifica se não existe nenhum jogo relevante.


        return 0.0
        # Nesse caso não existe o que recuperar.


    recommended = {game for game, _ in recommendations[:k]}
    # Pega somente os K jogos recomendados.


    return len(recommended & relevant) / len(relevant)
    # Calcula quantos relevantes foram encontrados
    # dividido pela quantidade total de relevantes.


def popular_recommendations(user_id, train_ratings, k=10, min_count=1):
    # Cria recomendações baseadas em popularidade.
    #
    # Esse modelo funciona como baseline para comparação.


    # Popularidade e calculada somente a partir das avaliacoes de treino.
    # Isso é importante para evitar que as avaliações do conjunto de teste
    # influenciem a recomendação.


    known = set(train_ratings.loc[train_ratings["user_id"] == user_id, "game_id"])
    # Encontra os jogos que o usuário já conhece/avaliou.
    #
    # .loc[...] seleciona as linhas daquele usuário.
    # Depois seleciona somente game_id.
    # set() transforma em conjunto.


    stats = train_ratings.groupby("game_id")["rating"].agg(["mean", "count"])
    # Agrupa as avaliações por jogo.
    #
    # Para cada jogo calcula:
    #
    # mean  -> média das avaliações
    # count -> quantidade de avaliações


    stats = stats[stats["count"] >= min_count].copy()
    # Remove jogos que possuem menos avaliações que o mínimo definido.


    stats["score"] = stats["mean"] + 0.01 * stats["count"]
    # Cria uma pontuação de popularidade.
    #
    # score = média + 0.01 × quantidade de avaliações
    #
    # A quantidade de avaliações funciona como um pequeno
    # desempate em favor de jogos mais avaliados.


    stats = stats.drop(index=known, errors="ignore").sort_values("score", ascending=False)
    # Remove os jogos que o usuário já conhece.
    #
    # errors="ignore" evita erro caso algum jogo de known não esteja
    # presente no índice.
    #
    # Depois ordena do maior score para o menor.


    return [(game, float(row.score)) for game, row in stats.head(k).iterrows()]
    # Pega os primeiros K jogos.
    #
    # iterrows() percorre as linhas.
    #
    # Retorna uma lista no formato:
    #
    # [(game_id, score), ...]


def evaluate_user(user_id, ratings, k=10, min_common_games=2, neighbor_count=10,
                  test_size=0.2, random_state=42, model="user"):
    # Avalia um modelo de recomendação para um usuário.
    #
    # model pode ser:
    # "user"
    # "item"
    # "popularity"


    # Todos os modelos recebem o mesmo split para manter a comparacao valida.
    # Isso significa que os modelos são avaliados sobre exatamente
    # a mesma divisão treino/teste.


    train_user, test_ratings = split_user_ratings(user_id, ratings, test_size, random_state)
    # Divide as avaliações do usuário em treino e teste.


    ratings_train = ratings.drop(test_ratings.index)
    # Remove do conjunto completo as avaliações que foram colocadas no teste.
    #
    # O restante vira o conjunto usado pelos modelos.


    matrix = create_user_game_matrix(ratings_train)
    # Cria a matriz usuário × jogo usando somente o treinamento.


    relevant = set(test_ratings.loc[test_ratings["rating"] >= 4, "game_id"])
    # Define como relevantes os jogos que o usuário avaliou com 4 ou 5
    # no conjunto de teste.


    neighbors = []
    # Inicializa a lista de vizinhos.
    #
    # Será preenchida somente para o modelo user-based.


    candidates = set()
    # Inicializa o conjunto de jogos candidatos.


    contributors = {}
    # Inicializa um dicionário que armazenará quantos vizinhos
    # contribuíram para cada recomendação.


    if model == "user" and user_id in matrix.index:
        # Entra aqui quando o modelo escolhido é user-based
        # e o usuário está presente na matriz.


        neighbors = find_similar_users(user_id, matrix, neighbor_count, min_common_games)
        # Encontra os usuários mais similares ao usuário-alvo.


        candidates = get_candidate_games(user_id, neighbors, matrix)
        # Encontra jogos avaliados pelos vizinhos que o usuário-alvo
        # ainda não avaliou.


        scores = calculate_game_scores(neighbors, candidates, matrix)
        # Calcula a pontuação prevista para cada jogo candidato.


        recommendations = rank_games(scores, k)
        # Ordena as pontuações e pega os primeiros K.


        contributors = {game: sum(pd.notna(matrix.loc[user, game]) for user, _, _ in neighbors)
                        for game in candidates}
        # Para cada jogo candidato conta quantos vizinhos possuem
        # uma avaliação para aquele jogo.
        #
        # pd.notna(...) verifica se a célula possui uma avaliação.
        #
        # O resultado fica aproximadamente:
        #
        # {
        #     jogo_1: 3,
        #     jogo_2: 1,
        #     jogo_3: 2
        # }


    elif model == "popularity":
        # Caso o modelo escolhido seja popularidade.


        recommendations = popular_recommendations(user_id, ratings_train, k)
        # Gera recomendações usando somente popularidade.


    elif model == "item":
        # Caso o modelo escolhido seja item-based.


        from src.recommendation.recommender import score_item_based_candidates
        # Importa a função responsável por calcular scores
        # baseados na similaridade entre jogos.


        scores = score_item_based_candidates(user_id, ratings_train)
        # Calcula a pontuação dos jogos candidatos.


        candidates = set(scores)
        # Pega os IDs dos jogos que receberam pontuação.


        recommendations = rank_games(scores, k)
        # Ordena os candidatos e seleciona os K primeiros.


    else:
        # Se model não for nenhum dos três valores esperados.


        raise ValueError("model invalido")
        # Interrompe a execução informando que o modelo é inválido.


    recommended_ids = {game for game, _ in recommendations}
    # Extrai somente os IDs dos jogos recomendados.


    return {
        # Retorna um dicionário contendo diversas métricas da avaliação.


        "user_id": user_id,
        # Identifica qual usuário foi avaliado.


        "total_ratings": len(ratings[ratings["user_id"] == user_id]),
        # Quantidade total de avaliações que o usuário possui.


        "train_ratings": len(train_user),
        # Quantidade de avaliações utilizadas no treinamento.


        "test_ratings": len(test_ratings),
        # Quantidade de avaliações reservadas para teste.


        "relevant_count": len(relevant),
        # Quantidade de jogos considerados relevantes.


        "neighbors": len(neighbors),
        # Quantidade de vizinhos encontrados.
        # Para popularity/item pode ser 0.


        "candidate_count": len(candidates),
        # Quantidade de jogos candidatos.


        "recommendation_count": len(recommendations),
        # Quantidade de recomendações realmente geradas.


        "intersection": sorted(recommended_ids & relevant),
        # Jogos que aparecem tanto nas recomendações
        # quanto nos jogos relevantes.


        "precision@5": precision_at_k(recommendations, relevant, 5),
        # Calcula Precision@5.


        "precision@10": precision_at_k(recommendations, relevant, 10),
        # Calcula Precision@10.


        "recall@5": recall_at_k(recommendations, relevant, 5),
        # Calcula Recall@5.


        "recall@10": recall_at_k(recommendations, relevant, 10),
        # Calcula Recall@10.


        "coverage": bool(recommendations),
        # True se pelo menos uma recomendação foi produzida.
        #
        # False se recommendations estiver vazio.


        "mean_contributors": (sum(contributors.values()) / len(contributors) if contributors else 0.0),
        # Calcula a média de contribuidores por jogo.
        #
        # Se contributors estiver vazio, retorna 0.0.


        "one_contributor_ratio": (sum(value == 1 for value in contributors.values()) / len(contributors)
                                  if contributors else 0.0),
        # Calcula a proporção dos candidatos que receberam contribuição
        # de exatamente 1 vizinho.


        "two_contributor_ratio": (sum(value == 2 for value in contributors.values()) / len(contributors)
                                  if contributors else 0.0),
        # Calcula a proporção dos candidatos que receberam contribuição
        # de exatamente 2 vizinhos.


        "three_plus_contributor_ratio": (sum(value >= 3 for value in contributors.values()) / len(contributors)
                                         if contributors else 0.0),
        # Calcula a proporção dos candidatos que receberam contribuição
        # de 3 ou mais vizinhos.


        "recommendations": recommendations,
        # Guarda a lista completa das recomendações.


        "relevant_games": sorted(relevant),
        # Guarda os jogos considerados relevantes,
        # ordenados pelos IDs.


        "contributors": contributors,
        # Guarda a quantidade de contribuidores por jogo.
    }


def evaluate_models(ratings, user_ids, min_common_games=2, neighbor_count=10,
                    test_size=0.2, random_state=42):
    # Avalia os três modelos para vários usuários.


    rows = []
    # Lista onde serão armazenados os resultados individuais.


    for model in ("popularity", "user", "item"):
        # Percorre os três modelos:
        #
        # popularity
        # user
        # item


        for user_id in user_ids:
            # Para cada modelo, percorre todos os usuários escolhidos.


            row = evaluate_user(user_id, ratings, 10, min_common_games, neighbor_count,
                                test_size, random_state, model)
            # Avalia aquele usuário usando aquele modelo.


            row["model"] = model
            # Adiciona ao resultado qual modelo foi utilizado.


            rows.append(row)
            # Adiciona o resultado à lista.


    details = pd.DataFrame(rows)
    # Transforma todos os resultados individuais em um DataFrame.


    summary = details.groupby("model").agg(
        # Agrupa os resultados por modelo e calcula médias.


        precision_at_5=("precision@5", "mean"),
        # Média da Precision@5.


        precision_at_10=("precision@10", "mean"),
        # Média da Precision@10.


        recall_at_5=("recall@5", "mean"),
        # Média do Recall@5.


        recall_at_10=("recall@10", "mean"),
        # Média do Recall@10.


        coverage=("coverage", "mean"),
        # Média de coverage.
        #
        # Como True = 1 e False = 0, a média funciona como
        # proporção de casos em que houve recomendação.


        mean_neighbors=("neighbors", "mean"),
        # Média da quantidade de vizinhos.


        mean_candidates=("candidate_count", "mean"),
        # Média da quantidade de candidatos.


        mean_recommendations=("recommendation_count", "mean"),
        # Média da quantidade de recomendações.


        mean_contributors=("mean_contributors", "mean"),
        # Média dos contribuidores.


        one_contributor_ratio=("one_contributor_ratio", "mean"),
        # Média da proporção de jogos com um contribuidor.


        two_contributor_ratio=("two_contributor_ratio", "mean"),
        # Média da proporção de jogos com dois contribuidores.


        three_plus_contributor_ratio=("three_plus_contributor_ratio", "mean"),
        # Média da proporção de jogos com três ou mais contribuidores.


    ).reset_index()
    # reset_index() transforma o índice "model" novamente em uma coluna.


    return summary, details
    # Retorna:
    #
    # summary -> resumo por modelo
    # details -> resultados individuais de cada usuário
