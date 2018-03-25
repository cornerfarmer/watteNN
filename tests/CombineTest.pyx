import unittest
from gym_watten.envs.watten_env cimport WattenEnv, Color, Value, Observation, Player, State
from src.ModelRating cimport ModelRating, HandCards
from src.LookUp cimport LookUp
from src.MCTS cimport MCTS, Storage
from src.Game cimport Game
from libcpp.vector cimport vector
import time

class CombinateTest(unittest.TestCase):

    def test_test(self):
        cdef WattenEnv env = WattenEnv()
        cdef LookUp model = LookUp()
        cdef LookUp best_model = LookUp()
        cdef Storage storage = Storage()
        cdef MCTS mcts = MCTS()
        cdef ModelRating rating = ModelRating(env)
        cdef Game game = Game(env)
        cdef int g
        cdef vector[float] eval_scores
        cdef float rating_value, exploitability

        for g in range(1000):
            mcts.mcts_generate(env, model, storage)

            model.memorize_storage(storage)

            rating_value = game.compare_given_games(model, best_model, rating)
            if rating_value > 0.5:
                best_model.table = model.table
                exploitability = rating.calc_exploitability(best_model)
                eval_scores.push_back(exploitability)
                print(exploitability)

