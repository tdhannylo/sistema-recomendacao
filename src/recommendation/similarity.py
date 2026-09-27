import pandas as pd
# Importa pandas.

from sklearn.metrics.pairwise import cosine_similarity
# Importa a implementação de similaridade de cosseno
# do scikit-learn.

def get_common_games(user_a, user_b, user_game_matrix):
    # Encontra os jogos que dois usuários avaliaram.


    user_a_ratings = user_game_matrix.loc[user_a]
    # Pega toda a linha do usuário A.


    user_b_ratings = user_game_matrix.loc[user_b]
    # Pega toda a linha do usuário B.


    common_games = user_a_ratings.notna() & user_b_ratings.notna()
    # Cria uma série de True/False.
    #
    # True significa:
    # A avaliou E B avaliou.
    #
    # & significa AND lógico elemento por elemento.


    return user_game_matrix.columns[common_games]
    # Retorna somente as colunas correspondentes aos jogos
    # que ambos avaliaram.

def get_common_ratings(user_a, user_b, user_game_matrix):
    # Retorna as avaliações dos dois usuários
    # somente nos jogos que ambos avaliaram.


    common_games = get_common_games(
        user_a,
        user_b,
        user_game_matrix
    )
    # Descobre os jogos em comum.


    user_a_ratings = user_game_matrix.loc[user_a, common_games]
    # Pega as avaliações do usuário A somente nesses jogos.


    user_b_ratings = user_game_matrix.loc[user_b, common_games]
    # Pega as avaliações do usuário B somente nesses jogos.


    return user_a_ratings, user_b_ratings
    # Retorna as duas listas/séries de avaliações.


def calculate_cosine_similarity(user_a, user_b, user_game_matrix):
    # Calcula a similaridade entre dois usuários.


    # A similaridade usa somente os jogos avaliados pelos dois usuarios.
    # Jogos que apenas um deles avaliou não entram no cálculo.


    user_a_ratings, user_b_ratings = get_common_ratings(
        user_a,
        user_b,
        user_game_matrix
    )
    # Obtém as avaliações nos jogos em comum.


    if len(user_a_ratings) == 0:
        # Verifica se não existe nenhum jogo em comum.


        return 0.0
        # Sem jogos em comum, não existe base para calcular similaridade.


    similarity = cosine_similarity(
        [user_a_ratings.values],
        [user_b_ratings.values]
    )
    # Calcula a similaridade de cosseno entre os dois vetores.
    #
    # Exemplo:
    #
    # usuário A = [5, 3, 4]
    # usuário B = [4, 3, 5]
    #
    # Esses vetores representam as avaliações
    # nos mesmos jogos.


    return similarity[0][0]
    # cosine_similarity retorna uma matriz.
    #
    # Como estamos comparando somente dois vetores,
    # o resultado de interesse está na posição [0][0].

def find_similar_users(
    target_user,
    user_game_matrix,
    k=5,
    min_common_games=2
):
    # Procura os K usuários mais similares ao usuário-alvo.
    #
    # min_common_games define quantos jogos os dois precisam
    # ter avaliado em comum.


    # Mantem os vizinhos que atingem o minimo de jogos em comum.
    # Isso evita considerar usuários com apenas uma pequena
    # quantidade de avaliações compartilhadas.


    similarities = []
    # Lista onde serão armazenados os usuários similares.


    for other_user in user_game_matrix.index:
        # Percorre todos os usuários existentes na matriz.


        if other_user == target_user:
            # Verifica se estamos comparando o usuário com ele mesmo.


            continue
            # Se for ele mesmo, pula para o próximo usuário.


        common_games = get_common_games(
            target_user,
            other_user,
            user_game_matrix
        )
        # Encontra os jogos avaliados pelos dois usuários.


        if len(common_games) < min_common_games:
            # Verifica se a quantidade de jogos em comum
            # é menor que o mínimo necessário.


            continue
            # Ignora esse usuário.


        similarity = calculate_cosine_similarity(
            target_user,
            other_user,
            user_game_matrix
        )
        # Calcula a similaridade de cosseno.


        similarities.append(
            (other_user, similarity, len(common_games))
        )
        # Armazena uma tupla contendo:
        #
        # ID do usuário
        # similaridade
        # quantidade de jogos em comum


    similarities.sort(
        key=lambda x: x[1],
        reverse=True
    )
    # Ordena os usuários pela similaridade.
    #
    # x[1] é o segundo elemento da tupla:
    # similarity.
    #
    # reverse=True -> maior primeiro.


    return similarities[:k]
    # Retorna somente os K usuários mais similares.


def build_item_similarity(ratings):
    # Calcula a similaridade entre todos os jogos.


    """Return item-item cosine similarities from the training ratings."""
    # Docstring explicando que a função retorna
    # similaridades jogo-jogo usando avaliações de treinamento.


    # Para item-based CF, cada linha representa um jogo e cada coluna um usuario.
    # Isso é o contrário da matriz usada no user-based.


    from scipy.sparse import csr_matrix
    # Importa csr_matrix, uma estrutura eficiente para matrizes esparsas.
    #
    # Como existem muitos valores ausentes/zero,
    # uma matriz esparsa pode economizar memória.


    from sklearn.metrics.pairwise import cosine_similarity as sparse_cosine
    # Importa cosine_similarity novamente,
    # mas usando o nome sparse_cosine para deixar claro
    # que será usada sobre a matriz esparsa.


    matrix = ratings.pivot(index="game_id", columns="user_id", values="rating").fillna(0)
    # Cria uma matriz jogo × usuário.
    #
    # Exemplo:
    #
    #            user_1 user_2 user_3
    # game_10       5      4      0
    # game_20       3      0      5
    #
    # fillna(0) transforma ausência de avaliação em 0.
    #
    # Isso é necessário para transformar a matriz em uma representação
    # numérica completa para o cálculo.


    similarities = sparse_cosine(csr_matrix(matrix.values))
    # Converte os valores da matriz para CSR sparse.
    #
    # Depois calcula a similaridade de cosseno entre todas as linhas.
    #
    # Como cada linha representa um jogo,
    # o resultado é uma similaridade jogo × jogo.


    return pd.DataFrame(similarities, index=matrix.index, columns=matrix.index)
    # Converte a matriz de similaridade em um DataFrame.
    #
    # As linhas representam jogos.
    # As colunas também representam jogos.
    #
    # Portanto:
    #
    # similarities.loc[jogo_A, jogo_B]
    #
    # representa o quanto A e B são similares.
