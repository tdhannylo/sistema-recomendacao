import pandas as pd
import unittest

from src.data.preprocess import create_user_game_matrix, validate_data
from src.evaluation.metrics import precision_at_k, recall_at_k, split_user_ratings
from src.recommendation.group_recommender import recommend_for_group
from src.recommendation.recommender import recommend_for_user
from src.recommendation.similarity import calculate_cosine_similarity, get_common_games


def sample_data():
    ratings = pd.DataFrame({
        "user_id": ["u1", "u1", "u1", "u2", "u2", "u2", "u3", "u3", "u3"],
        "game_id": [1, 2, 3, 1, 2, 4, 1, 3, 4],
        "rating": [5, 4, 1, 5, 4, 5, 5, 1, 4],
    })
    games = pd.DataFrame({"game_id": [1, 2, 3, 4], "name": list("ABCD")})
    return games, ratings


class BackendTests(unittest.TestCase):
    def test_data_and_similarity(self):
        games, ratings = sample_data()
        self.assertTrue(validate_data(games, ratings))
        matrix = create_user_game_matrix(ratings)
        self.assertEqual(list(get_common_games("u1", "u2", matrix)), [1, 2])
        self.assertGreater(calculate_cosine_similarity("u1", "u2", matrix), 0)


    def test_split_has_no_leakage(self):
        _, ratings = sample_data()
        train, test = split_user_ratings("u1", ratings, random_state=42)
        self.assertTrue(set(train.index).isdisjoint(test.index))
        self.assertTrue(set(train.game_id).isdisjoint(test.game_id))


    def test_metrics(self):
        recommendations = [(1, 5.0), (2, 4.0), (3, 3.0)]
        self.assertEqual(precision_at_k(recommendations, [2, 4], 2), 0.5)
        self.assertEqual(recall_at_k(recommendations, [2, 4], 2), 0.5)


    def test_user_recommendation_excludes_known_games(self):
        games, ratings = sample_data()
        result = recommend_for_user("u1", ratings, games, k=2, min_common_games=1, neighbor_count=2)
        self.assertLessEqual(len(result), 2)
        self.assertFalse({item["game_id"] for item in result} & set(ratings[ratings.user_id == "u1"].game_id))


    def test_group_requires_three_and_excludes_known_games(self):
        games, ratings = sample_data()
        with self.assertRaises(ValueError):
            recommend_for_group(["u1", "u2"], ratings, games, k=2,
                                min_common_games=1, neighbor_count=2)
        with self.assertRaises(ValueError):
            recommend_for_group(["u1", "u2", "u3", "u4"], ratings, games, k=2,
                                min_common_games=1, neighbor_count=2)
        result = recommend_for_group(["u1", "u2", "u3"], ratings, games, k=2,
                                     min_common_games=1, neighbor_count=2)
        self.assertLessEqual(len(result), 2)
        self.assertFalse({item["game_id"] for item in result} & set(ratings.game_id))


if __name__ == "__main__":
    unittest.main()
