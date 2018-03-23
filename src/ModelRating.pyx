from gym_watten.envs.watten_env cimport State, WattenEnv, Card, Observation
from src.LookUp cimport LookUp, ModelOutput
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.map cimport map
from libcpp.algorithm cimport sort
import itertools
from libc.string cimport memset

cdef extern from "<algorithm>" namespace "std" nogil:
    const T& max[T](const T& a, const T& b)

cdef extern from "<string>" namespace "std":
    string to_string(int val)

cdef class ModelRating:

    def __init__(self, WattenEnv env):
        self.env = env
        self._define_eval_games()

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
                self.eval_games.back().lastTrick.resize(2)
                self.eval_games.back().player0_tricks = 0
                self.eval_games.back().player1_tricks = 0

                for card_id in hand_cards:
                    self.eval_games.back().player0_hand_cards.push_back(self.env.cards[card_id])

                for card_id in other_hand_cards:
                    self.eval_games.back().player1_hand_cards.push_back(self.env.cards[card_id])

    cdef int search(self, Observation obs, LookUp model, vector[float]* all_values=NULL, int target_player=-1):
        cdef int current_player = self.env.current_player

        cdef State state = self.env.get_state()
        cdef int n = self.env.players[self.env.current_player].hand_cards.size()

        cdef int max_val = 0
        cdef int value
        cdef ModelOutput output

        if target_player is -1 or target_player == current_player:
            if target_player is -1:
                target_player = current_player

            for i in range(n):
                card_id = self.env.players[self.env.current_player].hand_cards[i].id
                self.env.step(card_id, &obs)

                if self.env.is_done():
                    value = self.env.last_winner == current_player
                else:
                    value = (1 - self.search(obs, model, NULL, target_player)) if current_player != self.env.current_player else self.search(obs, model, NULL, target_player)

                max_val = max(max_val, value)
                if all_values != NULL:
                    all_values.push_back(value)

                self.env.set_state(&state)

            return max_val
        else:
            model.predict_single(&obs, &output)

            self.env.step(model.valid_step(output.p, &self.env.players[self.env.current_player].hand_cards), &obs)

            if self.env.is_done():
                return self.env.last_winner == current_player
            else:
                return (1 - self.search(obs, model, NULL, target_player)) if current_player != self.env.current_player else self.search(obs, model, NULL, target_player)

    cdef string generate_hand_cards_key(self, vector[Card*]* hand_cards):
        cdef vector[int] card_ids
        for card in hand_cards[0]:
            card_ids.push_back(card.id)

        sort(card_ids.begin(), card_ids.end())

        cdef string t = <char*>"["
        cdef int card_id
        for card_id in card_ids:
            t += to_string(card_id) + <char*>","
        return t + <char*>"]"

    cdef string generate_cache_key(self, State* state):
        return self.generate_hand_cards_key(&state.player0_hand_cards if state.current_player == 0 else &state.player1_hand_cards) + <char*>"-" + self.generate_hand_cards_key(&state.player0_hand_cards if state.current_player == 1 else &state.player1_hand_cards) + <char*>"-" + (to_string(state.table_card.id) if state.table_card is not NULL else <char*>"") + <char*>"-" + to_string(state.player0_tricks if state.current_player == 0 else state.player1_tricks) + <char*>"-" + to_string(state.player0_tricks if state.current_player == 1 else state.player1_tricks)

    cdef vector[float] calc_correct_output_sample(self, State* state, LookUp model):
        cdef vector[Card*]* hand_cards = &state.player0_hand_cards if state.current_player == 0 else &state.player1_hand_cards
        cdef int i
        cdef vector[float] correct_output
        cdef Observation obs

        cache_key = self.generate_cache_key(state)
        if self.cache.count(cache_key) == 0:

            self.env.set_state(state)

            self.env.regenerate_obs(&obs)
            self.search(obs, model, &correct_output)

            i = 0
            for i in range(hand_cards.size()):
                self.cache[cache_key][hand_cards[0][i].id] = correct_output[i]
                i += 1

        cdef vector[float] output
        for i in range(hand_cards.size()):
            output.push_back(self.cache[cache_key][hand_cards[0][i].id])

        return output

    cdef vector[float] calc_correct_output(self, State state, LookUp model, vector[HandCards]* possible_hand_cards):
        cdef vector[float] correct_output
        cdef int i, c, n = 0

        for c in range(possible_hand_cards[0].size()):
            if state.current_player == 1:
                state.player0_hand_cards = possible_hand_cards[0][c]
            else:
                state.player1_hand_cards = possible_hand_cards[0][c]

            sample_outputs = self.calc_correct_output_sample(&state, model)

            if correct_output.size() == 0:
                correct_output = sample_outputs
            else:
                for i in range(correct_output.size()):
                    correct_output[i] += sample_outputs[i]

            n += 1

        for i in range(correct_output.size()):
            correct_output[i] /= n
        return correct_output

    cdef int calc_exploitability_in_game(self, LookUp model, vector[HandCards]* possible_hand_cards):
        cdef State state
        cdef int current_player, i, j, k
        cdef Observation obs
        cdef Card* card
        cdef ModelOutput output
        cdef int step

        current_player = self.env.current_player
        if self.env.current_player == 1:
            state = self.env.get_state()
            p = self.calc_correct_output(state, model, possible_hand_cards)
            self.env.set_state(&state)
            step = self.env.players[self.env.current_player].hand_cards[model.argmax(&p)].id
        else:
            self.env.regenerate_obs(&obs)
            model.predict_single(&obs, &output)

            step = model.valid_step(output.p, &self.env.players[self.env.current_player].hand_cards)

            i = 0
            while i < possible_hand_cards.size():

                memset(obs.hand_cards, 0, sizeof(obs.hand_cards))

                for card in possible_hand_cards[0][i]:
                    obs.hand_cards[<int>card.color][<int>card.value][0] = 1

                if self.env.table_card is not NULL:
                    obs.hand_cards[<int>self.env.table_card.color][<int>self.env.table_card.value][1] = 1

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

    cpdef float calc_exploitability(self, model):
        cdef vector[HandCards] opposite_hand_card_combinations
        cdef int exploitability = 0
        cdef Card* card
        self.cache.clear()

        for i in range(self.eval_games.size()):
            for start_player in range(2):
                self.env.set_state(&self.eval_games[i])
                self.env.current_player = start_player

                possible_opposite_cards = list(range(self.env.cards.size()))
                for card in self.env.players[1].hand_cards:
                    possible_opposite_cards.remove(card.id)

                combinations = itertools.combinations(possible_opposite_cards, self.env.players[0].hand_cards.size())

                opposite_hand_card_combinations.clear()
                for hand_cards in combinations:
                    opposite_hand_card_combinations.push_back(HandCards())
                    for hand_card in hand_cards:
                        opposite_hand_card_combinations.back().push_back(self.env.cards[hand_card])

                winner = ((1 - self.calc_exploitability_in_game(model, &opposite_hand_card_combinations)) if start_player == 1 else self.calc_exploitability_in_game(model, &opposite_hand_card_combinations))
                exploitability += (-1 if winner == 1 else 1)

        return <float>exploitability / (self.eval_games.size() * 2)
