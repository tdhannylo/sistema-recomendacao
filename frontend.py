from pathlib import Path
import pandas as pd
import streamlit as st

# Configurações de diretórios e arquivos
BASE_DIR = Path(__file__).resolve().parent
RATINGS_FILE = BASE_DIR / "data" / "raw" / "game_ratings.csv"
METADATA_FILE = BASE_DIR / "data" / "raw" / "games_metadata_5k.csv"

# Campos: game_id,user_id,rating


@st.cache_data
def load_data():
    # Carrega o arquivo CSV de avaliações de jogos, garantindo que os IDs de jogo e usuário sejam tratados como strings.
    return pd.read_csv(RATINGS_FILE, dtype={"game_id": str, "user_id": str})

# Campos: game_id,name,description,genres,platforms,rating,released,cover_image,game_link,metacritic_url


@st.cache_data
def load_metadata():
    # Carrega o arquivo CSV de metadados dos jogos, garantindo que os IDs de jogo sejam tratados como strings.
    metadata = pd.read_csv(METADATA_FILE, dtype={"game_id": str})
    # Remove duplicatas com base no campo "game_id" e cria uma cópia do DataFrame resultante.
    metadata = metadata.drop_duplicates("game_id").copy()
    # Preenche valores nulos de name com "Jogo sem nome"
    metadata["name"] = metadata["name"].fillna("Jogo sem nome")
    # cria um novo campo "label" que combina o nome do jogo com seu ID, formatado como "Zelda (#ID)"
    metadata["label"] = metadata["name"] + "  (#" + metadata["game_id"] + ")"
    return metadata

# Função de recomendação exemplo que retorna sempre os cinco jogos mais bem avaliados que não estão entre os selecionados.
# Futuramente, esta função será substituída por um algoritmo de recomendação baseado em distância entre perfis de jogadores.
# Talvez essa função tenha que ser alterada futuramente a depender do calculo de distância entre perfis de jogadores que for implementado. Por enquanto, ela apenas retorna os cinco jogos mais bem avaliados que não estão entre os selecionados.


def recommendation_preview(metadata, selected_game_ids):
    # Filtra os jogos disponíveis que não estão entre os IDs de jogos selecionados.
    available = metadata[~metadata["game_id"].isin(selected_game_ids)].copy()
    # Se a coluna "rating" estiver presente, converte os valores para numéricos
    # Ordena os jogos por avaliação em ordem decrescente e coloca valores nulos no final.
    if "rating" in available:
        available["rating"] = pd.to_numeric(
            available["rating"], errors="coerce")
        available = available.sort_values(
            "rating", ascending=False, na_position="last")
    # Retorna os cinco jogos mais bem avaliados disponíveis.
    return available.head(5)

# Apenas renderiza a prévia das 5 recomendações prontas com base nos IDs de jogos selecionados.
# Talvez essa função tenha que ser alterada futuramente a dependendo de como a função de recomendação real for implementada.


def render_recommendations(metadata, selected_game_ids):
    st.subheader("Sugestões para o grupo")
    # Mudar a legenda assim que a função de recomendação real estiver implementada.
    st.caption(
        "Prévia fictícia: futuramente, substitua esta função pelo cálculo de distância.")

    # Futuramente, esta função será substituída por um algoritmo de recomendação baseado em distância entre perfis de jogadores.
    recommendations = recommendation_preview(metadata, selected_game_ids)

    columns = st.columns(5)

    # Itera sobre as colunas e as recomendações, exibindo a imagem de capa, o nome e a data de lançamento de cada jogo recomendado.
    for column, (_, game) in zip(columns, recommendations.iterrows()):
        with column:
            if pd.notna(game.get("cover_image")) and game["cover_image"]:
                st.image(game["cover_image"], use_container_width=True)
            # Exibe o nome do jogo em negrito e, se disponível, a data de lançamento como legenda.
            st.markdown(f"**{game['name']}**")
            if pd.notna(game.get("released")):
                st.caption(str(game["released"]))

# Renderiza as preferências registradas de cada jogador e retorna a lista de IDs de jogos selecionados.


def render_group_preferences(selections, label_to_id):
    st.subheader("Preferências registradas")

    # Itera sobre as seleções de cada jogador, exibindo o nome do jogador e os jogos selecionados.
    for player_name, selected_labels in selections.values():
        st.markdown(f"**{player_name}**")
        st.write(", ".join(selected_labels))

    # Cria uma lista de IDs de jogos selecionados com base nos rótulos escolhidos pelos jogadores.
    selected_game_ids = []

    # Itera sobre as seleções de cada jogador, convertendo os rótulos dos jogos em IDs de jogos usando o dicionário label_to_id.
    for player_name, selected_labels in selections.values():
        for game_label in selected_labels:
            game_id = label_to_id[game_label]
            selected_game_ids.append(game_id)

    # Armazena os IDs de jogos selecionados no estado da sessão para uso posterior.
    st.session_state["selected_game_ids"] = selected_game_ids
    return selected_game_ids


def interface():
    # Wide evita que as colunas fiquem muito "espremidas", melhorando a visualização das recomendações lado a lado.
    st.set_page_config(page_title="Jogue em grupo",
                       page_icon="🎮", layout="wide")

    # O dataset de avaliações será usado posteriormente no cálculo de distância.
    metadata = load_metadata()

    # Carrega os dados de avaliações e metadados dos jogos.
    games = metadata[["game_id", "label"]].dropna().sort_values("label")
    game_labels = games["label"].tolist()
    label_to_id = dict(zip(games["label"], games["game_id"]))

    st.title("Encontre um jogo para jogar em grupo")
    st.write(
        "Monte o perfil de cada pessoa escolhendo cinco jogos que ela já conhece.")

    # Verifica se o processo de registro de jogadores já começou. Se não, exibe o formulário para o primeiro jogador.
    if not st.session_state.get("players_started", False):
        st.subheader("1. Quantos irão jogar?")

        # Permite que o usuário selecione a quantidade de jogadores, variando de 1 a 5.
        player_count = st.selectbox("Quantidade de jogadores",
                                    options=range(1, 6),
                                    format_func=lambda count: f"{count} jogador" if count == 1 else f"{count} jogadores",
                                    )

        # Formulário para o primeiro jogador registrar seu nome e selecionar cinco jogos que conhece.
        with st.form("first_player_preferences"):
            st.subheader("2. Suas preferências")
            player_name = st.text_input(
                "Seu nome",
                key="player_name_1",
                placeholder="Digite seu nome",
            )
            selected_labels = st.multiselect(
                "Escolha cinco jogos que você conhece",
                options=game_labels,
                max_selections=5,
                key="games_1",
                placeholder="Digite para buscar no catálogo...",
            )
            first_step_submitted = st.form_submit_button(
                "Continuar para outros jogadores" if player_count > 1 else "Continuar",
                type="primary",
                use_container_width=True,
            )

        # Tratamento de erros
        if not first_step_submitted:
            # Interrompe a execução se o usuário não clicou no botão de envio do formulário. (o botão de continuar)
            return
        if not player_name.strip():
            st.error("Informe seu nome.")
            return
        if len(selected_labels) != 5:
            st.error("Escolha exatamente cinco jogos antes de continuar.")
            return

        # Se houver apenas um jogador, registra as preferências e exibe as recomendações imediatamente.
        if player_count == 1:
            selections = {1: (player_name.strip(), selected_labels)}
            st.success("Preferências registradas com sucesso.")
            selected_game_ids = render_group_preferences(
                selections, label_to_id)
            render_recommendations(metadata, selected_game_ids)
            return

        # Se houver mais de um jogador, armazena as informações do primeiro jogador no estado da sessão
        # Também reinicia a execução para exibir o formulário dos outros jogadores.
        st.session_state["players_started"] = True
        st.session_state["player_count"] = player_count
        st.session_state["first_player_name"] = player_name.strip()
        st.session_state["first_player_games"] = list(selected_labels)
        # Recarrega a página e players_started será True, então o formulário para os outros jogadores será exibido.
        st.rerun()

    # Recupera o numero de jogadores escolhidos anteriormente
    player_count = st.session_state["player_count"]

    # Faz o mesmo que o formulário do primeiro jogador, mas para os outros jogadores, por meio do "for"
    with st.form("remaining_players_preferences"):
        st.subheader("3. Preferências dos outros jogadores")
        for player_number in range(2, player_count + 1):
            st.markdown(f"**Jogador {player_number}**")
            st.text_input(
                "Nome",
                key=f"player_name_{player_number}",
                placeholder=f"Digite o nome do jogador {player_number}",
            )
            st.multiselect(
                "Escolha cinco jogos",
                options=game_labels,
                max_selections=5,
                key=f"games_{player_number}",
                placeholder="Digite para buscar no catálogo...",
            )

        # Botão de envio do formulário para gerar as sugestões com base nas preferências registradas.
        suggestions_submitted = st.form_submit_button(
            "Gerar sugestões", type="primary", use_container_width=True)

    # Caso o usuário não tenha clicado no botão de envio do formulário, interrompe a execução.
    if not suggestions_submitted:
        return

    # Insere na posição 1 do dicionário selections as informações do primeiro jogador, que foram armazenadas no estado da sessão.
    selections = {
        1: (
            st.session_state.get("first_player_name", ""),
            st.session_state.get("first_player_games", []),
        )
    }

    # Recupera as informações dos outros jogadores a partir do estado da sessão e as adiciona ao dicionário selections.
    for player_number in range(2, player_count + 1):

        player_name = st.session_state.get(f"player_name_{player_number}", "")
        selected_labels = st.session_state.get(f"games_{player_number}", [])

        selections[player_number] = (player_name.strip(), selected_labels)

    # Validação da existencia de nomes e da quantidade de jogos selecionados para cada jogador.
    selected_names = []
    normalized_names = []

    for player_name, selected_labels in selections.values():
        selected_names.append(player_name)

        if not player_name:
            st.error("Informe o nome de cada jogador.")
            return

        if len(selected_labels) != 5:
            st.error(
                "Escolha exatamente cinco jogos para cada jogador antes de gerar as sugestões."
            )
            return

        # converte o nome do jogador para letras minúsculas
        normalized_name = player_name.casefold()
        normalized_names.append(normalized_name)

    # set() elimina nomes duplicados, se o tamanho for diferente então há nomes duplicados
    names_are_unique = len(set(normalized_names)) == len(selected_names)

    if not names_are_unique:
        st.error("Informe nomes diferentes para cada jogador.")
        return

    # Validação da quantidade de jogos selecionados para cada jogador.
    st.success("Preferências registradas com sucesso.")
    selected_game_ids = render_group_preferences(selections, label_to_id)
    render_recommendations(metadata, selected_game_ids)


def main():
    interface()


if __name__ == "__main__":
    main()
