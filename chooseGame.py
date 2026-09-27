import pandas as pd
import streamlit as st

from src.data.load_data import load_data
from src.recommendation.group_recommender import recommend_for_group


@st.cache_data
def get_processed_data():
    games, ratings = load_data(dataset="processed")
    return games.copy(), ratings.copy()


def build_game_label_map(games):
    games = games.dropna(subset=["name"]).copy()
    games["label"] = games.apply(
        lambda row: f"{row['name']} (#{int(row['game_id'])})",
        axis=1,
    )
    options = games.sort_values("name").reset_index(drop=True)
    return options, dict(zip(options["label"], options["game_id"]))


def find_best_user_for_player(selected_game_ids, ratings, used_user_ids=None):
    used_user_ids = set() if used_user_ids is None else {str(user_id) for user_id in used_user_ids}
    selected_game_ids = {int(game_id) for game_id in selected_game_ids}

    best_match = None
    for user_id in sorted(ratings["user_id"].astype(str).unique().tolist()):
        if user_id in used_user_ids:
            continue

        user_ratings = ratings[ratings["user_id"].astype(str) == user_id]
        matching = user_ratings[user_ratings["game_id"].isin(selected_game_ids)].copy()

        if matching.empty:
            continue

        coverage = int(matching["game_id"].nunique())
        total_rating = float(matching["rating"].sum())

        candidate = (coverage, total_rating, user_id)
        if best_match is None or candidate[0] > best_match[0] or (
            candidate[0] == best_match[0] and candidate[1] > best_match[1]
        ) or (
            candidate[0] == best_match[0] and candidate[1] == best_match[1] and candidate[2] < best_match[2]
        ):
            best_match = candidate

    if best_match is None:
        return None

    coverage, total_rating, user_id = best_match
    return {"user_id": user_id, "coverage": coverage, "total_rating": total_rating}


def render_recommendation_cards(recommendations):
    st.subheader("Sugestões para o grupo")
    rows = recommendations[:5]

    if not rows:
        st.warning("Nenhuma recomendação foi retornada para este grupo.")
        return

    columns = st.columns(min(len(rows), 5))

    for column, item in zip(columns, rows):
        with column:
            game_id = int(item["game_id"])
            metadata = st.session_state["game_metadata"].get(game_id)
            if metadata is not None:
                cover = metadata.get("cover_image")
                if pd.notna(cover) and str(cover).strip():
                    st.image(str(cover), use_container_width=True)
                st.markdown(f"**{metadata.get('name', f'Jogo {game_id}')}**")
                released = metadata.get("released")
                if pd.notna(released) and str(released).strip():
                    st.caption(str(released))
            else:
                st.markdown(f"**Jogo {game_id}**")
            st.caption(f"Score: {float(item.get('score', 0.0)):.2f}")


def interface():
    st.set_page_config(page_title="Jogue em grupo", page_icon="🎮", layout="wide")

    games, ratings = get_processed_data()
    game_options, label_to_id = build_game_label_map(games)
    game_labels = game_options["label"].tolist()

    if "players" not in st.session_state:
        st.session_state["players"] = []
    if "current_player" not in st.session_state:
        st.session_state["current_player"] = 0
    if "identified_players" not in st.session_state:
        st.session_state["identified_players"] = []
    if "recommendations" not in st.session_state:
        st.session_state["recommendations"] = None
    if "game_metadata" not in st.session_state:
        st.session_state["game_metadata"] = games.set_index("game_id").to_dict("index")

    st.title("Encontre um jogo para jogar em grupo")
    st.write("A aplicação trabalha com exatamente 3 jogadores, sequenciais e identificados por usuários reais do dataset processado.")
    st.subheader("Quantidade de jogadores: 3")

    if st.session_state["identified_players"] and st.session_state["recommendations"] is not None:
        st.subheader("Participantes identificados")
        for item in st.session_state["identified_players"]:
            st.write(f"{item['name']} → {item['user_id']} (cobertura: {item['coverage']}/3)")

        render_recommendation_cards(st.session_state["recommendations"])
        return

    current_player = st.session_state["current_player"]
    if current_player < 3:
        with st.form(f"player_form_{current_player}"):
            st.subheader(f"Jogador {current_player + 1}")
            player_name = st.text_input("Nome", key=f"player_name_{current_player}", placeholder="Digite o nome")
            selected_labels = st.multiselect(
                "Escolha até 3 jogos que conhece",
                options=game_labels,
                max_selections=3,
                key=f"player_games_{current_player}",
                placeholder="Digite para buscar no catálogo...",
            )

            button_label = "Continuar" if current_player < 2 else "Gerar recomendações"
            submitted = st.form_submit_button(button_label, type="primary", use_container_width=True)

        if not submitted:
            return

        if not player_name.strip():
            st.error("Informe o nome do jogador.")
            return

        if not selected_labels:
            st.error("Escolha pelo menos um jogo antes de continuar.")
            return

        selected_game_ids = [label_to_id[label] for label in selected_labels]
        players = st.session_state["players"]
        players.append({"name": player_name.strip(), "game_ids": selected_game_ids})
        st.session_state["players"] = players
        st.session_state["current_player"] = current_player + 1
        st.rerun()
        return

    players = st.session_state["players"]
    used_user_ids = []
    identified_players = []
    valid = True

    for participant in players:
        best_match = find_best_user_for_player(participant["game_ids"], ratings, used_user_ids)
        if best_match is None:
            st.error(f"Nenhum usuário foi encontrado para {participant['name']} com base nos jogos informados.")
            valid = False
            break

        while best_match["user_id"] in used_user_ids:
            used_user_ids.remove(best_match["user_id"])
            # in case of collision, we need a new candidate
            best_match = find_best_user_for_player(participant["game_ids"], ratings, used_user_ids)
            if best_match is None:
                st.error(f"Não foi possível associar um usuário diferente para {participant['name']}.")
                valid = False
                break

        if not valid:
            break

        used_user_ids.append(best_match["user_id"])
        identified_players.append({
            "name": participant["name"],
            "user_id": best_match["user_id"],
            "coverage": best_match["coverage"],
            "total_rating": best_match["total_rating"],
        })

    if not valid:
        return

    st.session_state["identified_players"] = identified_players
    user_ids = [player["user_id"] for player in identified_players]

    try:
        recommendations = recommend_for_group(user_ids, ratings, games, k=10, strategy="mean")
        st.session_state["recommendations"] = recommendations
    except Exception as exc:
        st.session_state["recommendations"] = []
        st.error(f"Erro ao gerar recomendações: {exc}")
        return

    st.subheader("Participantes identificados")
    for item in identified_players:
        st.write(f"{item['name']} → {item['user_id']} (cobertura: {item['coverage']}/3)")

    render_recommendation_cards(st.session_state["recommendations"])


def main():
    interface()


if __name__ == "__main__":
    main()