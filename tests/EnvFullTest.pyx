import unittest
from gym_watten.envs.watten_env cimport WattenEnv, Color, Value, Observation, Player, State, ActionType

class EnvFullTest(unittest.TestCase):

    def test_init(self):
        cdef WattenEnv env = WattenEnv()
        self.assertEqual(env.players.size(), 2, "wrong number of players")
        self.assertEqual(env.cards.size(), 32, "wrong number of cards")

        for i in range(env.cards.size()):
            for j in range(i + 1, env.cards.size()):
                self.assertFalse(env.cards[i].color == env.cards[j].color and env.cards[i].value == env.cards[j].value, "duplicate cards")

    def test_reset(self):
        cdef WattenEnv env = WattenEnv()
        cdef Observation obs
        env.reset()
        env.reset()
        env.reset()
        env.seed(1337)
        env.reset(&obs)

        cdef Player* player
        for player in env.players:
            self.assertEqual(player.hand_cards.size(), 5, "wrong number of hand cards")
            self.assertEqual(player.tricks, 0, "tricks not reseted")

        hand_cards = [[[0 for i in range(2)] for i in range(8)] for i in range(4)]
        hand_cards[0][6][0] = 1
        hand_cards[1][3][0] = 1
        hand_cards[2][5][0] = 1
        hand_cards[2][6][0] = 1
        hand_cards[3][7][0] = 1

        self.assertEqual(obs.sets, hand_cards, "wrong hand_cards obs")
        self.assertEqual(obs.scalars, [], "wrong tricks obs")
        self.assertEqual(obs.type, ActionType.CHOOSE_VALUE, "wrong action type obs")

        self.assertTrue(env.table_card == NULL, "wrong table card")
        self.assertEqual(env.last_tricks.size(), 0, "last tricks not empty")
        self.assertEqual(env.current_player, 0, "wrong current player")
        self.assertFalse(env.is_done(), "game is not done")

    def test_choose_value(self):
        cdef WattenEnv env = WattenEnv()
        cdef Observation obs
        env.seed(1337)
        env.reset()
        env.step(12, &obs)

        hand_cards = [[[0 for i in range(2)] for i in range(8)] for i in range(4)]
        hand_cards[0][3][0] = 1
        hand_cards[0][5][0] = 1
        hand_cards[2][4][0] = 1
        hand_cards[3][1][0] = 1
        hand_cards[3][4][0] = 1

        hand_cards[0][3][1] = 1
        hand_cards[1][3][1] = 1
        hand_cards[2][3][1] = 1
        hand_cards[3][3][1] = 1

        self.assertEqual(obs.sets, hand_cards, "wrong hand_cards obs")
        self.assertEqual(obs.scalars, [], "wrong tricks obs")
        self.assertEqual(obs.type, ActionType.CHOOSE_COLOR, "wrong action type obs")

        self.assertEqual(env.last_tricks.size(), 0, "last tricks not empty")
        self.assertTrue(env.chosen_value == Value.ZEHN, "wrong chosen value")
        self.assertEqual(env.current_player, 1, "wrong current player")
        self.assertFalse(env.is_done(), "game is not done")

    def test_choose_color(self):
        cdef WattenEnv env = WattenEnv()
        cdef Observation obs
        env.seed(1337)
        env.reset()
        env.step(12)
        env.step(3, &obs)

        hand_cards = [[[0 for i in range(12)] for i in range(8)] for i in range(4)]
        hand_cards[0][6][0] = 1
        hand_cards[1][3][0] = 1
        hand_cards[2][5][0] = 1
        hand_cards[2][6][0] = 1
        hand_cards[3][7][0] = 1

        hand_cards[0][3][10] = 1
        hand_cards[1][3][10] = 1
        hand_cards[2][3][10] = 1
        hand_cards[3][3][10] = 1

        hand_cards[0][0][11] = 1
        hand_cards[0][1][11] = 1
        hand_cards[0][2][11] = 1
        hand_cards[0][3][11] = 1
        hand_cards[0][4][11] = 1
        hand_cards[0][5][11] = 1
        hand_cards[0][6][11] = 1
        hand_cards[0][7][11] = 1

        self.assertEqual(obs.sets, hand_cards, "wrong hand_cards obs")
        self.assertEqual(obs.scalars, [0, 0, 0, 0], "wrong tricks obs")
        self.assertEqual(obs.type, ActionType.DRAW_CARD, "wrong action type obs")

        self.assertEqual(env.last_tricks.size(), 0, "last tricks not empty")
        self.assertTrue(env.chosen_value == Value.ZEHN, "wrong chosen value")
        self.assertTrue(env.chosen_color == Color.EICHEL, "wrong chosen color")
        self.assertEqual(env.current_player, 0, "wrong current player")
        self.assertFalse(env.is_done(), "game is not done")

    def test_step_win(self):
        cdef WattenEnv env = WattenEnv()
        cdef Observation obs
        env.last_winner = 1
        env.seed(1337)
        env.reset()
        env.step(12)
        env.step(3)

        env.step(12)
        env.step(4)

        env.step(30)
        env.step(24)

        env.step(18)
        env.step(2)

        env.step(19)
        env.step(1)

        env.step(17)
        env.step(27, &obs)

        self.assertEqual(obs.scalars, [1, 1, 0, 1], "wrong tricks obs")

        self.assertTrue(env.table_card == NULL, "wrong table card")
        self.assertEqual(env.current_player, 0, "wrong current player")
        self.assertTrue(env.is_done(), "game is done")
        self.assertEqual(env.last_winner, 0, "incorrect winner")
        self.assertEqual(env.invalid_move, False, "no invalid card")

