import unittest
from gym_watten.envs.watten_env cimport WattenEnv, Color, Value, Observation, Player, State
from src.ModelRating cimport ModelRating, HandCards
from src.LookUp cimport LookUp
from src.MCTS cimport MCTS, Storage
from src.Game cimport Game
from libcpp.vector cimport vector
import time

class ModelRatingTest(unittest.TestCase):

    def test_init(self):
        cdef WattenEnv env = WattenEnv()
        cdef ModelRating rating = ModelRating(env)

        self.assertEqual(rating.eval_games.size(), 560, "Wrong eval games number")
        for i in range(rating.eval_games.size()):
            self.assertEqual(rating.eval_games[i].player0_hand_cards.size(), 3, "Wrong hand card size in eval game (p0)")
            self.assertEqual(rating.eval_games[i].player1_hand_cards.size(), 3, "Wrong hand card size in eval game (p1)")
            #self.assertEqual(rating.eval_games[i].lastTrick.size(), 2, "Wrong lastTrick size in eval game")
            self.assertTrue(rating.eval_games[i].table_card == NULL, "Existing table card in eval game")
            self.assertEqual(rating.eval_games[i].player0_tricks, 0, "Wrong trick number in eval game (p0)")
            self.assertEqual(rating.eval_games[i].player1_tricks, 0, "Wrong trick number in eval game (p1)")
            self.assertEqual(rating.eval_games[i].current_player, 0, "Wrong current_player")

    def test_generate_calc_correct_output_sample(self):
        cdef WattenEnv env = WattenEnv()
        cdef ModelRating rating = ModelRating(env)
        cdef LookUp model = LookUp()

        env.seed(42)
        env.reset()

        cdef State state = env.get_state()

        cdef vector[float] p#' = rating.calc_correct_output_sample(&state, model)

        self.assertEqual(p, [1, 1, 0], "Wrong values")
        self.assertEqual(rating.cache["[3,4,7,]-[0,1,5,]--|0-0"], {3: 1.0, 4: 1.0, 7: 0.0}, "Wrong cache")


    def test_calc_correct_output(self):
        cdef WattenEnv env = WattenEnv()
        cdef ModelRating rating = ModelRating(env)
        cdef LookUp model = LookUp()

        env.seed(42)
        env.reset()
        env.current_player = 1

        cdef HandCards tmp = env.players[0].hand_cards
        env.players[0].hand_cards = env.players[1].hand_cards
        env.players[1].hand_cards = tmp

        cdef State state = env.get_state()
        cdef vector[HandCards] possible_hand_cards

        possible_hand_cards.push_back(HandCards())
        possible_hand_cards.back().push_back(env.cards[2])
        possible_hand_cards.back().push_back(env.cards[5])
        possible_hand_cards.back().push_back(env.cards[6])

        possible_hand_cards.push_back(HandCards(state.player0_hand_cards))

#        cdef vector[float] correct_output = rating.calc_correct_output(state, model, &possible_hand_cards)
     #   self.assertEqual(correct_output, [0.5, 1.0, 0.5], 'Wrong correct output')
    #    self.assertEqual(rating.cache.size(), 2, "Wrong cache number")

    def test_calc_exploitability_in_game(self):
        cdef WattenEnv env = WattenEnv()
        cdef ModelRating rating = ModelRating(env)
        cdef LookUp model = LookUp()

        env.seed(42)
        env.reset()

        cdef HandCards tmp = env.players[0].hand_cards
        env.players[0].hand_cards = env.players[1].hand_cards
        env.players[1].hand_cards = tmp
        cdef State state = env.get_state()

        cdef vector[HandCards] possible_hand_cards

        possible_hand_cards.push_back(HandCards())
        possible_hand_cards.back().push_back(env.cards[2])
        possible_hand_cards.back().push_back(env.cards[5])
        possible_hand_cards.back().push_back(env.cards[6])

        possible_hand_cards.push_back(HandCards(env.players[0].hand_cards))

        cdef int exploitability = rating.calc_exploitability_in_game(model, &possible_hand_cards)
        self.assertEqual(exploitability, 0, 'Wrong exploitability (0)')


        env.set_state(&state)
        env.current_player = 1

        possible_hand_cards.clear()
        possible_hand_cards.push_back(HandCards())
        possible_hand_cards.back().push_back(env.cards[2])
        possible_hand_cards.back().push_back(env.cards[5])
        possible_hand_cards.back().push_back(env.cards[6])

        possible_hand_cards.push_back(HandCards(env.players[0].hand_cards))

        exploitability = rating.calc_exploitability_in_game(model, &possible_hand_cards)
        self.assertEqual(exploitability, 1, 'Wrong exploitability (1)')

    def test_calc_exploitability(self):
        cdef WattenEnv env = WattenEnv()
        cdef ModelRating rating = ModelRating(env)
        cdef LookUp model = LookUp()

        self.assertAlmostEqual(rating.calc_exploitability(model), 0.5267857313156128, 5, "Wrong exploitability for empty model")
