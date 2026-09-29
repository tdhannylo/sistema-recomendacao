from statistics import mean

from src.data.load_data import load_data
from src.evaluation.metrics import evaluate_user

# Irá fazer o retorno de resultados dos dados dos testes, todos os testes

# CONFIGURAÇÃO

MODELS = ["user", "item", "popularity"]

K = 10
MIN_COMMON_GAMES = 2
NEIGHBOR_COUNT = 10
TEST_SIZE = 0.2
RANDOM_STATE = 42

NUMBER_OF_USERS = 5



# FUNÇÕES AUXILIARES

def get_common_users(ratings_raw, ratings_processed, number_of_users=5):
    """
    Retorna uma quantidade fixa de usuários presentes nos dois datasets.

    A ordenação garante que a escolha seja determinística.
    """

    raw_users = set(ratings_raw["user_id"].unique())
    processed_users = set(ratings_processed["user_id"].unique())

    common_users = sorted(raw_users & processed_users)

    if len(common_users) < number_of_users:
        raise ValueError(
            f"Não existem {number_of_users} usuários em comum "
            f"entre os datasets. Encontrados: {len(common_users)}."
        )

    return common_users[:number_of_users]


def evaluate_dataset(dataset_name, ratings, users):
    """
    Executa a avaliação dos três modelos para os usuários informados.
    """

    results = []

    for user_id in users:
        for model in MODELS:

            result = evaluate_user(
                user_id=user_id,
                ratings=ratings,
                k=K,
                min_common_games=MIN_COMMON_GAMES,
                neighbor_count=NEIGHBOR_COUNT,
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE,
                model=model,
            )

            result["dataset"] = dataset_name
            result["model"] = model

            results.append(result)

    return results


def average_metric(results, metric):
    """
    Calcula a média de uma métrica entre os resultados.
    """

    values = [
        result[metric]
        for result in results
        if metric in result
    ]

    return mean(values) if values else 0.0


def print_dataset_info(dataset_name, games, ratings):
    """
    Exibe informações básicas do dataset.
    """

    number_of_users = ratings["user_id"].nunique()
    number_of_games = ratings["game_id"].nunique()
    number_of_ratings = len(ratings)

    possible_interactions = number_of_users * number_of_games

    density = (
        number_of_ratings / possible_interactions
        if possible_interactions
        else 0
    )

    sparsity = 1 - density

    print("\n" + "=" * 70)
    print(f"DATASET: {dataset_name.upper()}")
    print("=" * 70)

    print(f"Usuários:       {number_of_users}")
    print(f"Jogos:          {number_of_games}")
    print(f"Avaliações:     {number_of_ratings}")
    print(f"Densidade:      {density:.4%}")
    print(f"Esparsidade:    {sparsity:.4%}")

    print(
        f"Média avaliações/usuário: "
        f"{number_of_ratings / number_of_users:.2f}"
    )

    print(
        f"Média avaliações/jogo:    "
        f"{number_of_ratings / number_of_games:.2f}"
    )


def print_individual_results(results, users):
    """
    Exibe os resultados individuais de cada usuário.
    """

    print("\n" + "=" * 70)
    print("RESULTADOS INDIVIDUAIS")
    print("=" * 70)

    for dataset_name in ["raw", "processed"]:

        print(f"\n--- DATASET: {dataset_name.upper()} ---")

        for user_id in users:

            print(f"\nUsuário: {user_id}")

            user_results = [
                result
                for result in results
                if result["dataset"] == dataset_name
                and result["user_id"] == user_id
            ]

            for result in user_results:

                print(f"\n  Modelo: {result['model']}")

                print(
                    f"    Avaliações totais: "
                    f"{result['total_ratings']}"
                )

                print(
                    f"    Avaliações treino: "
                    f"{result['train_ratings']}"
                )

                print(
                    f"    Avaliações teste: "
                    f"{result['test_ratings']}"
                )

                print(
                    f"    Relevantes (rating >= 4): "
                    f"{result['relevant_count']}"
                )

                if result["model"] == "user":

                    print(
                        f"    Vizinhos encontrados: "
                        f"{result['neighbors']}"
                    )

                    print(
                        f"    Candidatos: "
                        f"{result['candidate_count']}"
                    )

                    print(
                        f"    Média de contribuidores: "
                        f"{result['mean_contributors']:.4f}"
                    )

                    print(
                        f"    1 contribuidor: "
                        f"{result['one_contributor_ratio']:.2%}"
                    )

                    print(
                        f"    2 contribuidores: "
                        f"{result['two_contributor_ratio']:.2%}"
                    )

                    print(
                        f"    3+ contribuidores: "
                        f"{result['three_plus_contributor_ratio']:.2%}"
                    )

                elif result["model"] == "item":

                    print(
                        f"    Candidatos: "
                        f"{result['candidate_count']}"
                    )

                print(
                    f"    Recomendações: "
                    f"{result['recommendation_count']}"
                )

                print(
                    f"    Precision@5: "
                    f"{result['precision@5']:.4f}"
                )

                print(
                    f"    Precision@10: "
                    f"{result['precision@10']:.4f}"
                )

                print(
                    f"    Recall@5: "
                    f"{result['recall@5']:.4f}"
                )

                print(
                    f"    Recall@10: "
                    f"{result['recall@10']:.4f}"
                )

                print(
                    f"    Interseção recomendações/relevantes: "
                    f"{result['intersection']}"
                )


def print_model_comparison(results):
    """
    Exibe a média das métricas por modelo e dataset.
    """

    print("\n" + "=" * 70)
    print("COMPARAÇÃO DOS MODELOS")
    print("=" * 70)

    for dataset_name in ["raw", "processed"]:

        print(f"\n--- DATASET: {dataset_name.upper()} ---")

        for model in MODELS:

            model_results = [
                result
                for result in results
                if result["dataset"] == dataset_name
                and result["model"] == model
            ]

            if not model_results:
                continue

            print(f"\nModelo: {model}")

            print(
                f"  Precision@5:  "
                f"{average_metric(model_results, 'precision@5'):.4f}"
            )

            print(
                f"  Precision@10: "
                f"{average_metric(model_results, 'precision@10'):.4f}"
            )

            print(
                f"  Recall@5:     "
                f"{average_metric(model_results, 'recall@5'):.4f}"
            )

            print(
                f"  Recall@10:    "
                f"{average_metric(model_results, 'recall@10'):.4f}"
            )

            print(
                f"  Candidatos:   "
                f"{average_metric(model_results, 'candidate_count'):.2f}"
            )

            print(
                f"  Recomendações:"
                f" {average_metric(model_results, 'recommendation_count'):.2f}"
            )


def print_raw_vs_processed(results):
    """
    Compara diretamente raw e processed para cada modelo.
    """

    print("\n" + "=" * 70)
    print("RAW VS PROCESSED")
    print("=" * 70)

    for model in MODELS:

        raw_results = [
            result
            for result in results
            if result["dataset"] == "raw"
            and result["model"] == model
        ]

        processed_results = [
            result
            for result in results
            if result["dataset"] == "processed"
            and result["model"] == model
        ]

        print(f"\nModelo: {model}")

        if not raw_results or not processed_results:
            print("  Dados insuficientes para comparação.")
            continue

        print(
            f"  Precision@10:"
            f" raw={average_metric(raw_results, 'precision@10'):.4f}"
            f" | processed={average_metric(processed_results, 'precision@10'):.4f}"
        )

        print(
            f"  Recall@10:"
            f" raw={average_metric(raw_results, 'recall@10'):.4f}"
            f" | processed={average_metric(processed_results, 'recall@10'):.4f}"
        )

        print(
            f"  Candidatos:"
            f" raw={average_metric(raw_results, 'candidate_count'):.2f}"
            f" | processed={average_metric(processed_results, 'candidate_count'):.2f}"
        )

        if model == "user":

            print(
                f"  Média de contribuidores:"
                f" raw={average_metric(raw_results, 'mean_contributors'):.4f}"
                f" | processed={average_metric(processed_results, 'mean_contributors'):.4f}"
            )

            print(
                f"  1 contribuidor:"
                f" raw={average_metric(raw_results, 'one_contributor_ratio'):.2%}"
                f" | processed={average_metric(processed_results, 'one_contributor_ratio'):.2%}"
            )

            print(
                f"  2 contribuidores:"
                f" raw={average_metric(raw_results, 'two_contributor_ratio'):.2%}"
                f" | processed={average_metric(processed_results, 'two_contributor_ratio'):.2%}"
            )

            print(
                f"  3+ contribuidores:"
                f" raw={average_metric(raw_results, 'three_plus_contributor_ratio'):.2%}"
                f" | processed={average_metric(processed_results, 'three_plus_contributor_ratio'):.2%}"
            )


def print_detailed_first_user(results, user_id):
    """
    Exibe detalhes do primeiro usuário para facilitar a apresentação.
    """

    print("\n" + "=" * 70)
    print(f"DETALHAMENTO DO USUÁRIO: {user_id}")
    print("=" * 70)

    for dataset_name in ["raw", "processed"]:

        print(f"\n--- {dataset_name.upper()} ---")

        result = next(
            (
                item
                for item in results
                if item["dataset"] == dataset_name
                and item["user_id"] == user_id
                and item["model"] == "user"
            ),
            None,
        )

        if result is None:
            print("Resultado não encontrado.")
            continue

        print(
            f"Total de avaliações: {result['total_ratings']}"
        )

        print(
            f"Avaliações de treino: {result['train_ratings']}"
        )

        print(
            f"Avaliações de teste: {result['test_ratings']}"
        )

        print(
            f"Relevantes: {result['relevant_count']}"
        )

        print(
            f"Vizinhos: {result['neighbors']}"
        )

        print(
            f"Candidatos: {result['candidate_count']}"
        )

        print(
            f"Precision@5: {result['precision@5']:.4f}"
        )

        print(
            f"Precision@10: {result['precision@10']:.4f}"
        )

        print(
            f"Recall@5: {result['recall@5']:.4f}"
        )

        print(
            f"Recall@10: {result['recall@10']:.4f}"
        )

        print(
            f"Interseção: {result['intersection']}"
        )


# EXECUÇÃO

def main():

    print("=" * 70)
    print("AVALIAÇÃO DOS MODELOS DE RECOMENDAÇÃO")
    print("=" * 70)

    print("\nCarregando datasets...")

    games_raw, ratings_raw = load_data(dataset="raw")
    games_processed, ratings_processed = load_data(
        dataset="processed"
    )

    
    # Seleciona os mesmos 5 usuários para os dois datasets

    users = get_common_users(
        ratings_raw,
        ratings_processed,
        NUMBER_OF_USERS,
    )

    print("\nUsuários utilizados na avaliação:")

    for index, user_id in enumerate(users, start=1):
        print(f"  {index}. {user_id}")

    
    # Informações dos datasets

    print_dataset_info(
        "raw",
        games_raw,
        ratings_raw,
    )

    print_dataset_info(
        "processed",
        games_processed,
        ratings_processed,
    )

    # Avaliação

    print("\nExecutando avaliações...")

    raw_results = evaluate_dataset(
        "raw",
        ratings_raw,
        users,
    )

    processed_results = evaluate_dataset(
        "processed",
        ratings_processed,
        users,
    )

    results = raw_results + processed_results

    print("\nAvaliação concluída.")

    # Resultados

    print_individual_results(
        results,
        users,
    )

    print_model_comparison(
        results,
    )

    print_raw_vs_processed(
        results,
    )

    # Detalhamento do primeiro usuário

    print_detailed_first_user(
        results,
        users[0],
    )

    # Resumo final

    print("\n" + "=" * 70)
    print("RESUMO FINAL")
    print("=" * 70)

    print(
        f"\nUsuários avaliados: {len(users)}"
    )

    print(
        f"Usuários: {', '.join(users)}"
    )

    print(
        "\nConfiguração:"
    )

    print(
        f"  K recomendações: {K}"
    )

    print(
        f"  K vizinhos: {NEIGHBOR_COUNT}"
    )

    print(
        f"  Mínimo de jogos em comum: {MIN_COMMON_GAMES}"
    )

    print(
        f"  Test size: {TEST_SIZE}"
    )

    print(
        f"  Random state: {RANDOM_STATE}"
    )


if __name__ == "__main__":
    main()
