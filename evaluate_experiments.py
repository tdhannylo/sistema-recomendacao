import argparse
# Biblioteca para criar argumentos de linha de comando.
#
# Permite executar, por exemplo:
# python evaluate_experiments.py --dataset raw


import pandas as pd
# Importa pandas.


from src.data.load_data import load_data
# Importa a função central de carregamento dos datasets.


from src.data.preprocess import validate_data
# Importa a validação dos dados.


from src.evaluation.metrics import evaluate_models
# Importa a função que efetivamente executa os experimentos.


TEST_SIZE = 0.2
# 20% das avaliações serão usadas como teste.


RANDOM_STATE = 42
# Mantém a divisão treino/teste reproduzível.


MIN_COMMON_GAMES = 2
# Dois usuários precisam ter pelo menos 2 jogos em comum
# para serem considerados vizinhos.


NEIGHBOR_COUNT = 10
# Cada usuário terá no máximo 10 vizinhos.


MIN_USER_RATINGS = 10
# Para participar da avaliação, um usuário precisa
# ter pelo menos 10 avaliações.


def choose_users(ratings, minimum=MIN_USER_RATINGS, limit=20):
    # Seleciona usuários elegíveis para o experimento.


    counts = ratings.groupby("user_id").size()
    # Conta quantas avaliações cada usuário possui.


    return sorted(counts[counts >= minimum].index.astype(str))[:limit]
    # Mantém somente usuários com pelo menos "minimum" avaliações.
    #
    # .index -> pega os IDs.
    # .astype(str) -> garante que sejam strings.
    # sorted() -> ordena os IDs.
    # [:limit] -> pega no máximo 20.


def evaluate_dataset(dataset, selected_users=None):
    # Avalia um dataset específico:
    # raw ou processed.


    # Raw e processed sao carregados como pares consistentes de arquivos.
    # Garante que games e ratings pertençam ao mesmo dataset.


    games, ratings = load_data(dataset=dataset)
    # Carrega o dataset escolhido.


    validate_data(games, ratings)
    # Valida os dados.


    users = selected_users if selected_users is not None else choose_users(ratings)
    # Se selected_users foi fornecido:
    # usa exatamente esses usuários.
    #
    # Caso contrário:
    # escolhe automaticamente usuários elegíveis.


    if len(users) < 20:
        # Verifica se existem pelo menos 20 usuários.


        raise ValueError(f"O dataset {dataset} tem apenas {len(users)} usuarios elegiveis")
        # Interrompe se não houver usuários suficientes.


    summary, details = evaluate_models(
        ratings,
        users,
        min_common_games=MIN_COMMON_GAMES,
        neighbor_count=NEIGHBOR_COUNT,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    # Executa os experimentos.
    #
    # Os mesmos parâmetros são usados para manter a comparação
    # entre datasets/modelos consistente.


    summary.insert(0, "dataset", dataset)
    # Adiciona uma coluna "dataset" no início do DataFrame.
    #
    # Isso permite saber depois se uma linha pertence ao raw
    # ou ao processed.


    print(f"\nDATASET={dataset} | usuarios avaliados={len(users)} | "
          f"test_size={TEST_SIZE} | random_state={RANDOM_STATE}")
    # Mostra as configurações utilizadas.


    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    # Imprime o DataFrame de resultados.
    #
    # index=False -> não imprime o índice.
    #
    # float_format -> mostra valores decimais com 4 casas.


    return summary, details
    # Retorna:
    #
    # summary -> médias agregadas
    # details -> resultados individuais


def main():
    # Função principal do script.


    parser = argparse.ArgumentParser(
        description="Compara CF nos datasets raw e processed"
    )
    # Cria o parser dos argumentos da linha de comando.


    parser.add_argument(
        "--dataset",
        choices=("raw", "processed", "both"),
        default="both"
    )
    # Cria o argumento:
    #
    # --dataset raw
    # --dataset processed
    # --dataset both
    #
    # Se nada for informado, usa "both".


    args = parser.parse_args()
    # Lê os argumentos fornecidos pelo usuário.


    datasets = ("raw", "processed") if args.dataset == "both" else (args.dataset,)
    # Decide quais datasets serão avaliados.
    #
    # both -> ("raw", "processed")
    # raw -> ("raw",)
    # processed -> ("processed",)


    summaries = []
    # Lista que armazenará os resumos.


    selected_users = None
    # Inicialmente não existe uma lista específica de usuários.


    if args.dataset == "both":
        # Se a comparação for entre raw e processed...


        _, raw_ratings = load_data(dataset="raw")
        # Carrega somente ratings do raw.
        #
        # _ significa que games será ignorado.


        _, processed_ratings = load_data(dataset="processed")
        # Carrega ratings do processed.


        raw_users = set(
            choose_users(
                raw_ratings,
                limit=len(raw_ratings["user_id"].unique())
            )
        )
        # Obtém os usuários elegíveis do dataset raw.
        #
        # O limit grande faz com que não sejam limitados a apenas 20
        # neste momento.


        processed_counts = processed_ratings.groupby("user_id").size()
        # Conta quantas avaliações cada usuário possui no processed.


        selected_users = sorted(
            user for user in raw_users
            if user in processed_counts.index and processed_counts[user] >= MIN_USER_RATINGS
        )[:20]
        # Encontra usuários que existem nos DOIS datasets
        # e possuem pelo menos 10 avaliações no processed.
        #
        # Depois pega os primeiros 20.


        if len(selected_users) < 20:
            # Verifica se conseguiu encontrar 20 usuários comuns.


            raise ValueError("Nao ha 20 usuarios comuns com historico suficiente")
            # Interrompe se não houver.


        print(f"Usuarios comuns usados nos dois datasets: {len(selected_users)}")
        # Mostra quantos usuários foram selecionados.


    for dataset in datasets:
        # Percorre os datasets que devem ser avaliados.


        summary, _ = evaluate_dataset(dataset, selected_users)
        # Executa a avaliação.
        #
        # _ ignora o DataFrame detalhado.


        summaries.append(summary)
        # Guarda o resumo.


    if len(summaries) == 2:
        # Se os dois datasets foram avaliados...


        print("\nCOMPARACAO RAW VS PROCESSED")
        # Mostra um título.


        print(
            pd.concat(summaries, ignore_index=True).to_string(
                index=False,
                float_format=lambda value: f"{value:.4f}"
            )
        )
        # Junta os dois DataFrames de resumo
        # e imprime uma tabela única.


if __name__ == "__main__":
    main()
# Se esse arquivo for executado diretamente,
# chama a função main().
