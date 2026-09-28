# Biblioteca para criar argumentos de linha de comando como python evaluate_experiments.py --dataset raw
import argparse
import pandas as pd
from src.data.load_data import load_data
from src.data.preprocess import validate_data
from src.evaluation.metrics import evaluate_models

TEST_SIZE = 0.2
RANDOM_STATE = 42
# Dois usuários precisam ter pelo menos 2 jogos em comum para serem considerados vizinhos
MIN_COMMON_GAMES = 2
# Cada usuário terá no máximo 10 vizinhos
NEIGHBOR_COUNT = 10
# Para participar da avaliação, um usuário precisa ter pelo menos 10 avaliações
MIN_USER_RATINGS = 10

# Seleciona usuários elegíveis para o experimento
def choose_users(ratings, minimum=MIN_USER_RATINGS, limit=20):
    # Conta quantas avaliações cada usuário possui
    counts = ratings.groupby("user_id").size()
    # Mantém somente usuários com pelo menos uma quantidade mínima de avaliações, index sendo seus ids
    return sorted(counts[counts >= minimum].index.astype(str))[:limit]
    
# Avalia um dataset específico
def evaluate_dataset(dataset, selected_users=None):
    # Carrega o dataset escolhido
    games, ratings = load_data(dataset=dataset)
    # Valida os dados
    validate_data(games, ratings)
    # Se selected_users foi fornecido, usa estes senão outros usuários elegíveis
    users = selected_users if selected_users is not None else choose_users(ratings)
    # Verifica se existem pelo menos 20 usuários
    if len(users) < 20:
        raise ValueError(f"O dataset {dataset} tem apenas {len(users)} usuarios elegiveis")

    summary, details = evaluate_models(
        ratings,
        users,
        min_common_games=MIN_COMMON_GAMES,
        neighbor_count=NEIGHBOR_COUNT,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    # Isso permite saber depois se uma linha pertence ao raw ou processed
    summary.insert(0, "dataset", dataset)
    # Mostra as configurações utilizadas
    print(f"\nDATASET={dataset} | usuarios avaliados={len(users)} | "
          f"test_size={TEST_SIZE} | random_state={RANDOM_STATE}")
    

    # Imprime o DataFrame de resultados
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    # Retorna médias e resultados individuais
    return summary, details
    

def main():
    # Cria o parser dos argumentos da linha de comando
    parser = argparse.ArgumentParser(
        description="Compara CF nos datasets raw e processed"
    )
    
    # Cria o argumento --dataset raw, --dataset processed e --dataset both
    parser.add_argument(
        "--dataset",
        choices=("raw", "processed", "both"),
        default="both"
    )
    # Lê os argumentos fornecidos pelo usuário
    args = parser.parse_args()
    # Decide quais datasets serão avaliados
    datasets = ("raw", "processed") if args.dataset == "both" else (args.dataset,)
    # Lista que armazenará os resumos
    summaries = []
    # Inicialmente não existe uma lista específica de usuários
    selected_users = None

    if args.dataset == "both":
        # _ significa que games será ignorado.
        _, raw_ratings = load_data(dataset="raw")
        _, processed_ratings = load_data(dataset="processed")

        # Obtém os usuários elegíveis do dataset raw
        raw_users = set(
            choose_users(
                raw_ratings,
                limit=len(raw_ratings["user_id"].unique())
            )
        )

        # Conta quantas avaliações cada usuário possui no processed
        processed_counts = processed_ratings.groupby("user_id").size()
        
        # Encontra usuários que existem nos DOIS datasets e possuem pelo menos 10 avaliações no processed, depois pega os primeiros 20
        selected_users = sorted(
            user for user in raw_users
            if user in processed_counts.index and processed_counts[user] >= MIN_USER_RATINGS
        )[:20]

        # Verifica se conseguiu encontrar 20 usuários comuns.
        if len(selected_users) < 20:
            raise ValueError("Nao ha 20 usuarios comuns com historico suficiente")
        print(f"Usuarios comuns usados nos dois datasets: {len(selected_users)}")

    # Percorre os datasets que devem ser avaliados.
    for dataset in datasets:
        # Executa a avaliação e ignora o DataFrame avaliado com _ 
        summary, _ = evaluate_dataset(dataset, selected_users)
        # Guarda o resumo.
        summaries.append(summary)
        
    # Se os dois datasets foram avaliados
    if len(summaries) == 2:   
        print("\nCOMPARACAO RAW VS PROCESSED")
        # Junta os dois DataFrames de resumo e imprime uma tabela única
        print(
            pd.concat(summaries, ignore_index=True).to_string(
                index=False,
                float_format=lambda value: f"{value:.4f}"
            )
        )

if __name__ == "__main__":
    main()
