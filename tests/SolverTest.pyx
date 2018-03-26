import unittest
from gym_watten.envs.watten_env cimport WattenEnv, Color, Value, Observation, Player, State, Card
from src.ModelRating cimport ModelRating, HandCards
from src.Solver cimport Solver
from src cimport ModelOutput
from libcpp.vector cimport vector

class SolverTest(unittest.TestCase):

    def test_solve(self):
        cdef WattenEnv env = WattenEnv()
        cdef ModelRating rating = ModelRating(env)
        cdef Solver solver = Solver(env)
        cdef vector[Card*] tmp
        solver.solve(rating)
        solver.model.watch = True

        #rating.eval_games.erase(rating.eval_games.begin(), rating.eval_games.begin() + 210)
        #rating.eval_games.resize(10)
        #for i in range(rating.eval_games.size()):
        #    tmp = rating.eval_games[i].player0_hand_cards
        #    rating.eval_games[i].player0_hand_cards = rating.eval_games[i].player1_hand_cards
        #    rating.eval_games[i].player1_hand_cards = tmp
        #    print([card.id for card in rating.eval_games[i].player0_hand_cards], [card.id for card in rating.eval_games[i].player1_hand_cards])

        print([card.id for card in rating.eval_games[70].player0_hand_cards], [card.id for card in rating.eval_games[70].player1_hand_cards])
        env.set_state(&rating.eval_games[70])
        env.current_player = 1
        env.step(4)
        cdef Observation obs
        cdef ModelOutput output
        env.regenerate_obs(&obs)
        solver.model.predict_single(&obs, &output)
        print(output.p)

        print(rating.calc_exploitability(solver.model))
