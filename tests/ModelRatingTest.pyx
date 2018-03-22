import unittest
from gym_watten.envs.watten_env cimport WattenEnv, Color, Value, Observation, Player, State
from src.ModelRating cimport ModelRating
from libcpp.vector cimport vector

class ModelRatingTest(unittest.TestCase):

    def test_init(self):
        cdef WattenEnv env = WattenEnv()
        cdef ModelRating rating = ModelRating(env)

        self.assertEqual(rating.eval_games.size(), 560, "Wrong eval games number")
        for i in range(rating.eval_games.size()):
            self.assertEqual(rating.eval_games[i].player0_hand_cards.size(), 3, "Wrong hand card size in eval game (p0)")
            self.assertEqual(rating.eval_games[i].player1_hand_cards.size(), 3, "Wrong hand card size in eval game (p1)")
            self.assertEqual(rating.eval_games[i].lastTrick.size(), 2, "Wrong lastTrick size in eval game")
            self.assertTrue(rating.eval_games[i].table_card == NULL, "Existing table card in eval game")
            self.assertEqual(rating.eval_games[i].player0_tricks, 0, "Wrong trick number in eval game (p0)")
            self.assertEqual(rating.eval_games[i].player1_tricks, 0, "Wrong trick number in eval game (p1)")

    def test_search(self):
        cdef WattenEnv env = WattenEnv()
        cdef ModelRating rating = ModelRating(env)
        cdef Observation obs

        env.seed(42)
        env.reset(&obs)

        cdef vector[float] values
        print(rating.search(obs, &values))

        print(values)
