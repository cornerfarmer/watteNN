from gym_watten.envs.watten_env cimport State, WattenEnv, Card, Observation, ActionType
from src.LookUp cimport Model, LookUp, ModelOutput
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.map cimport map
from libcpp.algorithm cimport sort
import itertools
from libcpp cimport bool
from libc.string cimport memset
from libc.stdlib cimport rand
import sys

cdef extern from "<algorithm>" namespace "std" nogil:
    const T& max[T](const T& a, const T& b)

cdef extern from "<string>" namespace "std":
    string to_string(int val)

cdef class ModelRating:

    def __init__(self, WattenEnv env):
        self.env = env
        self._define_eval_games()
        self.memory = LookUp()

    cdef void _define_eval_games(self):
        self.eval_games.clear()
        hand_card_combinations = itertools.combinations(list(range(self.env.cards.size())), 3)
        for hand_cards in hand_card_combinations:
            cards_left = list(range(self.env.cards.size()))
            for card in hand_cards:
                cards_left.remove(card)

            other_hand_card_combinations = itertools.combinations(cards_left, 3)
            for other_hand_cards in other_hand_card_combinations:
                self.eval_games.push_back(State())
                self.eval_games.back().current_player = 0
                self.eval_games.back().table_card = NULL
                #self.eval_games.back().lastTrick.resize(2)
                self.eval_games.back().player0_tricks = 0
                self.eval_games.back().player1_tricks = 0
                self.eval_games.back().type = ActionType.DRAW_CARD if self.env.minimal else ActionType.CHOOSE_VALUE

                for card_id in hand_cards:
                    self.eval_games.back().player0_hand_cards.push_back(self.env.cards[card_id])

                for card_id in other_hand_cards:
                    self.eval_games.back().player1_hand_cards.push_back(self.env.cards[card_id])

    cdef vector[float] calc_correct_output_sample(self, State* state, Model model, vector[HandCards]* possible_hand_cards):
        cdef vector[Card*]* hand_cards = &state.player0_hand_cards if state.current_player == 0 else &state.player1_hand_cards
        cdef int i
        cdef vector[float] correct_output
        cdef Observation obs
        cdef vector[HandCards] next_possible_handcards
       # cache_key = self.generate_cache_key(state)
       # if True or self.cache.count(cache_key) == 0:
        self.env.set_state(state)

        for i in range(self.env.players[1].hand_cards.size()):
            card_id = self.env.players[1].hand_cards[i].id
            self.env.step(card_id, &obs)

            next_possible_handcards = possible_hand_cards[0]

            if self.env.is_done():
                correct_output.push_back(self.env.last_winner == 1)
            else:
                correct_output.push_back((1 - self.calc_exploitability_in_game(model, &next_possible_handcards)) if 1 != self.env.current_player else self.calc_exploitability_in_game(model, &next_possible_handcards))

            self.env.set_state(state)

       # for i in range(hand_cards.size()):
         #   self.cache[cache_key][hand_cards[0][i].id] = correct_output[i]

       # cdef vector[float] output
       # for i in range(hand_cards.size()):
       #     output.push_back(self.cache[cache_key][hand_cards[0][i].id])
#
        return correct_output

    cdef int calc_correct_output(self, State state, Model model, vector[HandCards]* possible_hand_cards):
        cdef vector[float] correct_output
        cdef int i, c, n = 0
        cdef Observation obs
        cdef ModelOutput model_output
        self.env.regenerate_obs(&obs)

        if not self.memory.is_memorized(&obs):
            for c in range(possible_hand_cards.size()):
                state.player0_hand_cards = possible_hand_cards[0][c]

                sample_outputs = self.calc_correct_output_sample(&state, model, possible_hand_cards)

                if correct_output.size() == 0:
                    correct_output = sample_outputs
                else:
                    for i in range(correct_output.size()):
                        correct_output[i] += sample_outputs[i]

                n += 1

            for i in range(32):
                model_output.p[i] = 0
            model_output.v = 0

            for i in range(correct_output.size()):
                model_output.p[self.env.players[self.env.current_player].hand_cards[i].id] = correct_output[i] / n

            #print([card.id for card in self.env.players[self.env.current_player].hand_cards], obs.hand_cards)
            self.memory.memorize(&obs, &model_output)
        else:
            self.memory.predict_single(&obs, &model_output)

        return self.memory.valid_step(model_output.p, &state.player1_hand_cards)

    cdef int calc_exploitability_in_game(self, Model model, vector[HandCards]* possible_hand_cards):
        cdef State state
        cdef int current_player, i, j, k
        cdef Observation obs
        cdef Card* card
        cdef ModelOutput output
        cdef int step


        current_player = self.env.current_player
        if self.env.current_player == 1:
            state = self.env.get_state()
            step = self.calc_correct_output(state, model, possible_hand_cards)

            self.env.set_state(&state)
        else:
            self.env.regenerate_obs(&obs)
            model.predict_single(&obs, &output)

            step = model.valid_step(output.p, &self.env.players[self.env.current_player].hand_cards)

            i = 0
            while i < possible_hand_cards.size():

                for j in range(4):
                    for k in range(8):
                        obs.sets[j][k][0] = 0

                for card in possible_hand_cards[0][i]:
                    obs.sets[<int>card.color][<int>card.value][0] = 1

                model.predict_single(&obs, &output)

                theoretical_step = model.valid_step(output.p, &possible_hand_cards[0][i])
                if step == theoretical_step:
                    for j in range(possible_hand_cards[0][i].size()):
                        if possible_hand_cards[0][i][j].id == step:
                            possible_hand_cards[0][i].erase(possible_hand_cards[0][i].begin() + j)
                            break
                else:
                    possible_hand_cards.erase(possible_hand_cards.begin() + i)
                    i -= 1
                i += 1
        self.env.step(step)


        if self.env.is_done():
            return self.env.last_winner == current_player
        else:
            return (1 - self.calc_exploitability_in_game(model, possible_hand_cards)) if current_player != self.env.current_player else self.calc_exploitability_in_game(model, possible_hand_cards)

    cpdef float calc_exploitability(self, Model model):
        cdef vector[HandCards] opposite_hand_card_combinations
        cdef int exploitability = 0
        cdef Card* card
        self.memory.table.clear()

        for i in range(self.eval_games.size()):
            for start_player in range(2):
                self.env.set_state(&self.eval_games[i])
                self.env.current_player = start_player

                possible_opposite_cards = []
                for card in self.env.cards:
                    possible_opposite_cards.append(card.id)

                for card in self.env.players[1].hand_cards:
                    possible_opposite_cards.remove(card.id)

                combinations = itertools.combinations(possible_opposite_cards, self.env.players[0].hand_cards.size())

                opposite_hand_card_combinations.clear()
                for hand_cards in combinations:
                    opposite_hand_card_combinations.push_back(HandCards())
                    for hand_card in hand_cards:
                        opposite_hand_card_combinations.back().push_back(self.env.all_cards[hand_card])

                winner = ((1 - self.calc_exploitability_in_game(model, &opposite_hand_card_combinations)) if start_player == 1 else self.calc_exploitability_in_game(model, &opposite_hand_card_combinations))
               # if start_player == 1 and self.eval_games[i].player0_hand_cards[0].id == 1 and self.eval_games[i].player0_hand_cards[1].id == 3 and self.eval_games[i].player0_hand_cards[2].id == 5:
               #     print(i, winner)
                exploitability += (-1 if winner == 1 else 1)

        return <float>exploitability / (self.eval_games.size() * 2)


    cpdef float calc_exploitability_by_random_games(self, Model model, int number_of_games):
        cdef vector[HandCards] opposite_hand_card_combinations
        cdef int start_player, exploitability = 0
        cdef Card* card
        self.memory.table.clear()

        for i in range(number_of_games):
            self.env.reset()
            start_player = rand() % 2
            self.env.current_player = start_player

            possible_opposite_cards = []
            for card in self.env.cards:
                possible_opposite_cards.append(card.id)
            for card in self.env.players[1].hand_cards:
                possible_opposite_cards.remove(card.id)

            combinations = itertools.combinations(possible_opposite_cards, self.env.players[0].hand_cards.size())

            opposite_hand_card_combinations.clear()
            for hand_cards in combinations:
                opposite_hand_card_combinations.push_back(HandCards())
                for hand_card in hand_cards:
                    opposite_hand_card_combinations.back().push_back(self.env.all_cards[hand_card])

            winner = ((1 - self.calc_exploitability_in_game(model, &opposite_hand_card_combinations)) if start_player == 1 else self.calc_exploitability_in_game(model, &opposite_hand_card_combinations))
           # if start_player == 1 and self.eval_games[i].player0_hand_cards[0].id == 1 and self.eval_games[i].player0_hand_cards[1].id == 3 and self.eval_games[i].player0_hand_cards[2].id == 5:
           #     print(i, winner)
            exploitability += (-1 if winner == 1 else 1)

        return <float>exploitability / number_of_games

    cpdef find(self, first_player, second_player):
        for i in range(self.eval_games.size()):
            self.env.set_state(&self.eval_games[i])

            first_key = ""
            for card in self.env.players[0].hand_cards:
                first_key += self.env.filename_from_card(self.env.all_cards[card.id]).decode('utf-8') + ","

            second_key = ""
            for card in self.env.players[1].hand_cards:
                second_key += self.env.filename_from_card(self.env.all_cards[card.id]).decode('utf-8') + ","

            if first_key == first_player and second_key == second_player:
                return i
        return -1

