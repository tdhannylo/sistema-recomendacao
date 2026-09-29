# Sistema de Recomendação de Jogos

## Sobre o projeto

Este é um projeto de um sistema de recomendação de jogos.

A ideia é utilizar as avaliações feitas por diferentes usuários para encontrar jogos que possam ser interessantes para uma pessoa ou para um grupo de jogadores.

O sistema possui recomendações individuais (`chooseGame.py`) e recomendações para um grupo (`chooseUser.py`) de **três jogadores**.

Para as recomendações em grupo, jogos que já foram avaliados por algum dos participantes não são apresentados novamente.

## Objetivo

O projeto tem como objetivo demonstrar o funcionamento de um sistema de recomendação utilizando avaliações de usuários.

O sistema busca:

* utilizar avaliações de jogos;
* identificar usuários com preferências semelhantes;
* gerar recomendações;
* evitar jogos que o usuário já avaliou;
* combinar recomendações de três jogadores;
* comparar diferentes formas de gerar recomendações;
* avaliar os resultados obtidos.

## Dados utilizados

O projeto utiliza um conjunto de dados com informações sobre jogos e avaliações de usuários.

Cada avaliação possui:

* um usuário;
* um jogo;
* uma nota de 1 a 5.

### Dataset original

Os arquivos originais estão em `data/raw/`:

* `games_metadata_5k.csv` — informações sobre os jogos;
* `game_ratings.csv` — avaliações dos usuários.

O dataset possui:

| Informação | Quantidade |
| ---------- | ---------: |
| Usuários   |     10.000 |
| Jogos      |      5.000 |
| Avaliações |    273.669 |

Como existem muitos jogos e relativamente poucas avaliações para cada usuário, os dados são bastante dispersos.

### Dataset processado

Também foi criado um conjunto de dados em `data/processed/` para os experimentos do projeto.

Esse conjunto utiliza usuários e jogos que possuem uma quantidade maior de avaliações. O critério utilizado é o percentil 80.

As avaliações originais não são alteradas nem criadas durante esse processo.

| Informação | Original | Processado |
| ---------- | -------: | ---------: |
| Usuários   |   10.000 |      2.042 |
| Jogos      |    5.000 |      1.009 |
| Avaliações |  273.669 |     23.978 |

O dataset processado é utilizado para os experimentos e para a interface atual do sistema.

## Como as recomendações são geradas

O projeto utiliza **filtragem colaborativa**.

De forma simplificada, o sistema procura identificar padrões nas avaliações dos usuários. Quando usuários possuem comportamentos semelhantes, as avaliações de um podem ajudar a encontrar jogos que possam interessar ao outro.

Foram utilizadas três formas de recomendação:

### Recomendação baseada em usuários

É a principal abordagem do projeto.

O sistema procura usuários que possuem avaliações semelhantes às do usuário analisado. Os jogos avaliados por esses usuários podem então ser utilizados para gerar novas recomendações.

### Recomendação baseada em jogos

Também foi utilizada uma abordagem que considera a relação entre os próprios jogos.

Ela serve principalmente como forma de comparação com a recomendação baseada em usuários.

### Popularidade

Também existe uma recomendação baseada na popularidade dos jogos.

Ela funciona como uma referência simples para comparar os resultados das outras abordagens.

## Recomendação para grupos

A recomendação para grupos (`chooseUser.py`) trabalha com **exatamente três jogadores**.

O processo é:

1. identificar os três usuários;
2. gerar recomendações para cada usuário;
3. retirar jogos que já foram avaliados por algum participante;
4. combinar as recomendações;
5. apresentar os jogos com maior pontuação para o grupo.

O sistema possui duas formas de combinar as recomendações:

* **Média:** combina as pontuações dos participantes;
* **Mínimo:** considera a menor pontuação disponível para o jogo.

A interface atual utiliza a opção **Média**.

## Recomendação Individual

A recomendação individual (`chooseGame.py`) trabalha com jogadores que participam de forma sequencial:

1. o primeiro jogador informa seu nome e escolhe de 1 a 3 jogos que conhece;
2. o segundo jogador realiza o mesmo procedimento;
3. o terceiro jogador realiza o mesmo procedimento;
4. o sistema associa cada participante a um usuário existente no dataset;
5. as recomendações são geradas para os três usuários;
6. os cinco primeiros resultados são apresentados na tela.

O nome informado pelo participante é utilizado apenas para identificação na interface. As recomendações são geradas a partir dos usuários existentes no dataset.

## Avaliação

O projeto também possui testes para verificar o funcionamento das recomendações.

Durante a avaliação, parte das avaliações de um usuário é separada. O sistema tenta recomendar os jogos retirados e os resultados são comparados com essas avaliações.

São consideradas relevantes as avaliações com notas **4 ou 5**.

As principais métricas utilizadas são:

* **Precision@K:** verifica quantas das primeiras recomendações são relevantes;
* **Recall@K:** verifica quantos dos jogos relevantes foram encontrados;
* **Coverage:** verifica para quantos usuários o sistema conseguiu gerar recomendações.

## Resultados dos experimentos

Foram realizados experimentos utilizando o dataset original e o dataset processado.

No protocolo utilizado, as métricas de Precision e Recall permaneceram em zero para os modelos avaliados.

| Dataset    | Modelo       | Precision@5 | Precision@10 | Recall@5 | Recall@10 | Coverage |
| ---------- | ------------ | ----------: | -----------: | -------: | --------: | -------: |
| Original   | User-based   |           0 |            0 |        0 |         0 |        1 |
| Original   | Item-based   |           0 |            0 |        0 |         0 |        1 |
| Original   | Popularidade |           0 |            0 |        0 |         0 |        1 |
| Processado | User-based   |           0 |            0 |        0 |         0 |        1 |
| Processado | Item-based   |           0 |            0 |        0 |         0 |        1 |
| Processado | Popularidade |           0 |            0 |        0 |         0 |        1 |

Os experimentos indicam que a quantidade e a distribuição das avaliações dificultam a obtenção de recomendações bem avaliadas no protocolo utilizado.

O dataset processado possui uma maior concentração de dados, mas isso não resultou em melhoria nas métricas de Precision e Recall.

## Limitações

O projeto possui algumas limitações:

* existem poucas avaliações em comparação com a quantidade total de jogos;
* muitos usuários possuem poucos jogos avaliados em comum;
* a quantidade de informações disponíveis dificulta encontrar preferências semelhantes;
* os resultados de Precision e Recall foram zero nos experimentos realizados;
* o dataset processado não apresentou melhoria nas métricas;

## Tecnologias utilizadas

* Python
* pandas
* NumPy
* SciPy
* scikit-learn
* Streamlit

## Como executar

### 1. Criar o dataset processado

```text
python create_processed_dataset.py
```

### 2. Executar os testes

```text
python -m unittest -v test_system.py
```

### 3. Executar os experimentos

Para comparar os dois datasets:

```text
python evaluate_experiments.py --dataset both
```

Para avaliar somente o dataset original:

```text
python evaluate_experiments.py --dataset raw
```

Para avaliar somente o dataset processado:

```text
python evaluate_experiments.py --dataset processed
```

Para avaliar o retorno dos resultados dos testes:

```text
python show_test_results.py 
```

### 4. Executar a interface principal

```text
python -m streamlit run chooseGame.py
```
```text
python -m streamlit run chooseUser.py
```

## Estrutura do projeto

```text
sistema-recomendacao/
│
├── data/
│   ├── processed/
│   └── raw/
│
│── pages/
│   └── history.py
│
├── src/
│   ├── data/
│   ├── evaluation/
│   └── recommendation/
│       
├── chooseGame.py
├── chooseUser.py
├── create_processed_dataset.py
├── evaluate_experiments.py
├── README.md
├── show_test_results.py
└── test_system.py
```
