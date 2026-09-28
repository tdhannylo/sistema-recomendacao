import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Encontra os jogos que dois usuários avaliaram
def get_common_games(user_a, user_b, user_game_matrix):
    # Pega toda a linha do usuário A
    user_a_ratings = user_game_matrix.loc[user_a]
    user_b_ratings = user_game_matrix.loc[user_b]

    # Cria uma série de True/False em que True significa A avaliou e B avaliou
    common_games = user_a_ratings.notna() & user_b_ratings.notna()

    # Retorna somente as colunas correspondentes aos jogos que ambos avaliaram
    return user_game_matrix.columns[common_games]
    
# Retorna as avaliações dos dois usuários somente nos jogos que ambos avaliaram
def get_common_ratings(user_a, user_b, user_game_matrix):
    # Descobre os jogos em comum
    common_games = get_common_games(
        user_a,
        user_b,
        user_game_matrix
    )
    # Pega as avaliações do usuário A somente nesses jogos
    user_a_ratings = user_game_matrix.loc[user_a, common_games]
    user_b_ratings = user_game_matrix.loc[user_b, common_games]
    # Retorna as duas listas/séries de avaliações
    return user_a_ratings, user_b_ratings
    

# Calcula a similaridade entre dois usuários, somente com os jogos avaliados pelos dois
def calculate_cosine_similarity(user_a, user_b, user_game_matrix):
    # Obtém as avaliações nos jogos em comum
    user_a_ratings, user_b_ratings = get_common_ratings(
        user_a,
        user_b,
        user_game_matrix
    )
    # Verifica se não existe nenhum jogo em comum
    if len(user_a_ratings) == 0:
        # Sem jogos em comum, não existe base para calcular similaridade
        return 0.0 

    # Calcula a similaridade de cosseno entre os dois vetores
    similarity = cosine_similarity(
        [user_a_ratings.values],
        [user_b_ratings.values]
    )
    # cosine_similarity retorna uma matriz, em que o resultado está na primeira posição da matriz
    return similarity[0][0]

# Procura os K usuários mais similares ao usuário-alvo
def find_similar_users(
    target_user,
    user_game_matrix,
    k=5,
    min_common_games=2
):
    
    #
    # min_common_games define quantos jogos os dois precisam
    # ter avaliado em comum.


    # Mantem os vizinhos que atingem o minimo de jogos em comum.
    # Isso evita considerar usuários com apenas uma pequena
    # quantidade de avaliações compartilhadas.

    # Lista onde serão armazenados os usuários similares
    similarities = []
    

    # Percorre todos os usuários existentes na matriz
    for other_user in user_game_matrix.index:
        # Verifica se estamos comparando o usuário com ele mesmo
        if other_user == target_user: 
            # Se for ele mesmo, pula para o próximo usuário
            continue

        # Encontra os jogos avaliados pelos dois usuários
        common_games = get_common_games(
            target_user,
            other_user,
            user_game_matrix
        )

        # Verifica se a quantidade de jogos em comum é menor que o mínimo necessário
        if len(common_games) < min_common_games:
            # Ignora esse usuário
            continue  

        # Calcula a similaridade de cosseno
        similarity = calculate_cosine_similarity(
            target_user,
            other_user,
            user_game_matrix
        )

        # Armazena uma tupla contendo user_id. similaridade e quant de jogos em comum
        similarities.append(
            (other_user, similarity, len(common_games))
        )

    # Ordena os usuários pela similaridade a partir do segundo elemento
    similarities.sort(
        key=lambda x: x[1],
        reverse=True
    )
    # Retorna somente os K usuários mais similares
    return similarities[:k]
    
# Calcula a similaridade entre todos os jogos em que linha é um jogo e coluna um usuário
def build_item_similarity(ratings):
    # Como existem muitos valores ausentes/zero, uma matriz esparsa pode economizar memória
    from scipy.sparse import csr_matrix
    from sklearn.metrics.pairwise import cosine_similarity as sparse_cosine

    # Cria uma matriz jogo × usuário em que ausência de avaliação é 0 em vez de NaN
    matrix = ratings.pivot(index="game_id", columns="user_id", values="rating").fillna(0)
    # Converte os valores da matriz para CSR sparse, depois calcula a similaridade de cosseno entre todas as linhas, resultado é uma similaridade jogo x jogo
    similarities = sparse_cosine(csr_matrix(matrix.values))
    # Converte a matriz de similaridade em um DataFrame
    return pd.DataFrame(similarities, index=matrix.index, columns=matrix.index)
