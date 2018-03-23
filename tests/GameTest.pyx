import unittest
from gym_watten.envs.watten_env cimport WattenEnv, Color, Value, Observation, Player, State
from src.Game cimport Game
from src.LookUp cimport LookUp
from src.ModelRating cimport ModelRating
from libcpp.vector cimport vector
import time

class GameTest(unittest.TestCase):

    def test_match(self):
        cdef WattenEnv env = WattenEnv()
        cdef Game game = Game(env)
        cdef LookUp model1 = LookUp()
        cdef LookUp model2 = LookUp()

        env.seed(42)
        self.assertEqual(game.match(model1, model2), 0)

    def test_compare(self):
        cdef WattenEnv env = WattenEnv()
        cdef Game game = Game(env)
        cdef LookUp model1 = LookUp()
        cdef LookUp model2 = LookUp()
        cdef ModelRating rating = ModelRating(env)

        env.seed(42)
        self.assertEqual(game.compare_given_games(model1, model2, rating), 0.5)