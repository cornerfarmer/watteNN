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
import re

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
            self.memory.predict_single_p(&obs, &model_output)

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
            model.predict_single_p(&obs, &output)

            step = model.valid_step(output.p, &self.env.players[self.env.current_player].hand_cards)

            i = 0
            while i < possible_hand_cards.size():

                for j in range(4):
                    for k in range(8):
                        obs.sets[j][k][0] = 0

                for card in possible_hand_cards[0][i]:
                    obs.sets[<int>card.color][<int>card.value][0] = 1

                model.predict_single_p(&obs, &output)

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

    cpdef find(self, key):
        m = [listing.strip(',') for listing in key.split('-')]

        own_cards = m[1].split(',')
        if len(own_cards) == 1 and own_cards[0] == '':
            own_cards = []
        opponent_cards = []

        last_tricks = m[0].split(',')
        if len(last_tricks) == 1 and last_tricks[0] == '':
            last_tricks = []

        for i in range(len(last_tricks)):
            last_tricks[i] = last_tricks[i].split('.')
            if last_tricks[i][1] == '0':
                own_cards.append(last_tricks[i][0])
            else:
                opponent_cards.append(last_tricks[i][0])

        if len(last_tricks) > 0:
            current_player = int(last_tricks[0][1])
        else:
            current_player = 0

        if current_player == 1:
            own_cards, opponent_cards = opponent_cards, own_cards
            for i in range(len(last_tricks)):
                last_tricks[i][1] = '1' if last_tricks[i][1] is '0' else '0'

        own_cards = [int(card_id) for card_id in own_cards]
        opponent_cards = [int(card_id) for card_id in opponent_cards]

        missing_cards = []
        cdef Card* card
        for card in self.env.cards:
            if not card.id in own_cards and not card.id in opponent_cards:
                missing_cards.append(card.id)

        combinations = itertools.combinations(missing_cards, 3 - min(len(opponent_cards), len(own_cards)))

        print("Current: " + str(current_player))
        print("P0: " + str(own_cards))
        print("P1: " + str(opponent_cards))
        print("Missing: " + str(missing_cards) + "\n")

        for combination in combinations:
            for i in range(self.eval_games.size()):
                valid = True

                for card in self.eval_games[i].player0_hand_cards:
                    if not card.id in own_cards and (current_player is 0 or not card.id in combination):
                        valid = False

                for card in self.eval_games[i].player1_hand_cards:
                    if not card.id in opponent_cards and (current_player is 1 or not card.id in combination):
                        valid = False

                if valid:
                    readable_own = ""
                    index = 0
                    for card in self.eval_games[i].player0_hand_cards:
                        readable_own += self.env.filename_from_card(self.env.all_cards[card.id]).decode('utf-8') + ","
                        for last_trick in last_tricks:
                            if last_trick[1] == '0' and int(last_trick[0]) == card.id:
                                last_trick.append(index)
                        index += 1

                    readable_opponent = ""
                    index = 0
                    for card in self.eval_games[i].player1_hand_cards:
                        readable_opponent += self.env.filename_from_card(self.env.all_cards[card.id]).decode('utf-8') + ","
                        for last_trick in last_tricks:
                            if last_trick[1] == '1' and int(last_trick[0]) == card.id:
                                last_trick.append(index)
                        index += 1

                    for j in range(len(last_tricks)):
                        for k in range(j + 1, len(last_tricks)):
                            if last_tricks[j][1] == last_tricks[k][1] and last_tricks[j][2] < last_tricks[k][2]:
                                last_tricks[k][2] -= 1

                    print(str(i) + " -> P0: " + readable_own + " - P1:" + readable_opponent + " - " + str([last_trick[2] for last_trick in last_tricks]))
                    break

            if not valid:
                print("Did not found: " + str(combination))


