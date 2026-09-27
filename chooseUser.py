import pandas as pd
import streamlit as st

from src.data.load_data import load_data
from src.recommendation.group_recommender import recommend_for_group


@st.cache_data
def get_processed_data():
    games, ratings = load_data(dataset="processed")
    return games.copy(), ratings.copy()


def build_group_recommendation_frame(user_ids, ratings, games, k=10):
    recommendations = recommend_for_group(
        user_ids,
        ratings,
        games,
        k=k,
        strategy="mean",
    )

    if not recommendations:
        return pd.DataFrame(columns=["game_id", "name", "cover_image", "released", "score"])

    metadata = games.set_index("game_id", drop=False).copy()
    rows = []

    for item in recommendations:
        game_id = int(item["game_id"])
        metadata_row = metadata.loc[metadata["game_id"] == game_id]

        if metadata_row.empty:
            rows.append({
                "game_id": game_id,
                "name": f"Jogo {game_id}",
                "cover_image": "",
                "released": pd.NA,
                "score": float(item.get("score", 0.0)),
            })
            continue

        info = metadata_row.iloc[0]
        rows.append({
            "game_id": game_id,
            "name": info.get("name", f"Jogo {game_id}"),
            "cover_image": info.get("cover_image", ""),
            "released": info.get("released", pd.NA),
            "score": float(item.get("score", 0.0)),
        })

    result = pd.DataFrame(rows)
    if not result.empty:
        result["score"] = pd.to_numeric(result["score"], errors="coerce").fillna(0.0)
        result = result.sort_values("score", ascending=False).head(5).reset_index(drop=True)

    return result


def render_recommendation_cards(recommendations):
    st.subheader("Sugestões para o grupo")

    if recommendations.empty:
        st.warning("Nenhuma recomendação foi retornada para este grupo.")
        return

    columns = st.columns(min(len(recommendations), 5))

    for column, row in zip(columns, recommendations.itertuples(index=False)):
        with column:
            cover = getattr(row, "cover_image", "")
            if pd.notna(cover) and str(cover).strip():
                st.image(str(cover), use_container_width=True)

            st.markdown(f"**{row.name}**")
            released = getattr(row, "released", pd.NA)
            if pd.notna(released) and str(released).strip():
                st.caption(str(released))
            st.caption(f"Score: {float(row.score):.2f}")


def interface():
    st.set_page_config(page_title="Jogue em grupo", page_icon="🎮", layout="wide")

    games, ratings = get_processed_data()
    user_ids = sorted(ratings["user_id"].dropna().astype(str).unique().tolist())

    if not user_ids:
        st.error("Nenhum usuário disponível no dataset processado.")
        return

    st.title("Encontre um jogo para jogar em grupo")
    st.write("Selecione exatamente três usuários para receber recomendações via o backend real.")

    with st.form("group_selection"):
        col1, col2, col3 = st.columns(3)
        with col1:
            user_1 = st.selectbox(
                "Jogador 1",
                options=user_ids,
                index=0,
                key="user_1",
            )
        with col2:
            user_2 = st.selectbox(
                "Jogador 2",
                options=user_ids,
                index=min(1, len(user_ids) - 1),
                key="user_2",
            )
        with col3:
            user_3 = st.selectbox(
                "Jogador 3",
                options=user_ids,
                index=min(2, len(user_ids) - 1),
                key="user_3",
            )

        submitted = st.form_submit_button("Gerar recomendações", type="primary", use_container_width=True)

    selected_users = [user_1, user_2, user_3]

    if submitted:
        if len(set(selected_users)) != 3:
            st.error("Selecione três usuários diferentes antes de gerar as recomendações.")
            return

        st.session_state["selected_users"] = selected_users

        try:
            recommendations = build_group_recommendation_frame(selected_users, ratings, games, k=10)
            st.session_state["recommendations"] = recommendations
        except Exception as exc:
            st.session_state["recommendations"] = pd.DataFrame()
            st.error(f"Erro ao gerar recomendações: {exc}")

    selected_users = st.session_state.get("selected_users", selected_users)
    if selected_users:
        st.subheader("Grupo selecionado")
        for index, user_id in enumerate(selected_users, start=1):
            st.write(f"Usuário {index}: {user_id}")

    recommendations = st.session_state.get("recommendations", pd.DataFrame())
    if recommendations is not None and not recommendations.empty:
        render_recommendation_cards(recommendations)
    elif "selected_users" in st.session_state:
        st.warning("Nenhuma recomendação foi retornada para os usuários selecionados.")


def main():
    interface()


if __name__ == "__main__":
    main()