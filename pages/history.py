import streamlit as st
from src.data.load_data import load_data

st.set_page_config(
    page_title="Histórico de avaliações",
    page_icon="🎮",
    layout="wide",
)


@st.cache_data
def load_processed_data():
    return load_data(dataset="processed")


games, ratings = load_processed_data()


st.title("Histórico de avaliações")
st.write("Consulte os jogos já avaliados por um usuário e as respectivas avaliações.")


users = sorted(ratings["user_id"].unique())


selected_user = st.selectbox(
    "Selecione um usuário",
    users,
)


user_ratings = ratings[
    ratings["user_id"] == selected_user
].copy()


user_history = user_ratings.merge(
    games[["game_id", "name", "cover_image", "released"]],
    on="game_id",
    how="left",
)


user_history = user_history.sort_values(
    by=["rating", "name"],
    ascending=[False, True],
)


st.subheader(f"Histórico de {selected_user}")

st.write(
    f"Total de jogos avaliados: **{len(user_history)}**"
)


if user_history.empty:
    st.info("Este usuário não possui avaliações no dataset.")
else:
    for _, game in user_history.iterrows():
        col_image, col_info = st.columns([1, 5])

        with col_image:
            if game["cover_image"]:
                st.image(
                    game["cover_image"],
                    width=100,
                )

        with col_info:
            st.markdown(f"### {game['name']}")

            rating = int(game["rating"])

            st.write(
                f"**Avaliação:** {'★' * rating}{'☆' * (5 - rating)} "
                f"({rating}/5)"
            )

            if game["released"]:
                st.write(f"**Lançamento:** {game['released']}")