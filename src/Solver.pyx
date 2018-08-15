from gym_watten.envs.watten_env cimport State, WattenEnv, Card, Observation
from src.LookUp cimport LookUp, ModelOutput
from src.ModelRating cimport ModelRating, HandCards
from libcpp.vector cimport vector
from libcpp cimport bool
import itertools

cdef class Solver:

    def __init__(self, WattenEnv env):
        self.env = env
        self.model = LookUp()

    cdef float search(self, Observation obs, vector[HandCards]* possible_handcards):
        cdef int current_player = self.env.current_player
        cdef vector[HandCards] next_possible_handcards

        cdef State state = self.env.get_state()
        cdef int n = self.env.players[self.env.current_player].hand_cards.size()

        cdef float max_val = 0
        cdef float value
        cdef ModelOutput output
        cdef bool valid_hand_cards
        cdef int i, j, k, c

        if current_player == 0:
            state = self.env.get_state()
            for i in range(n):
                card_id = self.env.players[0].hand_cards[i].id
                self.env.step(card_id, &obs)

                next_possible_handcards = possible_handcards[0]

                j = 0
                while j < next_possible_handcards.size():
                    valid_hand_cards = False

                    for k in range(next_possible_handcards[j].size()):
                        if next_possible_handcards[j][k].id == card_id:
                            next_possible_handcards[j].erase(next_possible_handcards[j].begin() + k)
                            valid_hand_cards = True
                            break

                    if not valid_hand_cards:
                        next_possible_handcards.erase(next_possible_handcards.begin() + j)
                        j -= 1
                    j += 1

                if self.env.is_done():
                    value = self.env.last_winner == current_player
                else:
                    value = (1 - self.search(obs, &next_possible_handcards)) if current_player != self.env.current_player else self.search(obs, &next_possible_handcards)

                max_val = max(value, max_val)

                self.env.set_state(&state)

            return max_val
        else:

            self.env.step(self.choose_step(obs, possible_handcards), &obs)

            if self.env.is_done():
                return self.env.last_winner == current_player
            else:
                return (1 - self.search(obs, possible_handcards)) if current_player != self.env.current_player else self.search(obs, possible_handcards)


    cdef int choose_step(self, Observation obs, vector[HandCards]* possible_handcards):
        cdef State state
        cdef State state_org
        cdef vector[float] correct_output
        cdef ModelOutput model_output, tmp
        if not self.model.is_memorized(&obs):

            state_org = self.env.get_state()

            for i in range(self.env.players[self.env.current_player].hand_cards.size()):
                correct_output.push_back(0)

            for c in range(possible_handcards.size()):
                self.env.players[0].hand_cards = possible_handcards[0][c]

                state = self.env.get_state()

                for i in range(self.env.players[self.env.current_player].hand_cards.size()):
                    card_id = self.env.players[self.env.current_player].hand_cards[i].id
                    self.env.step(card_id, &obs)

                    if self.env.is_done():
                        correct_output[i] += (self.env.last_winner == 1)
                    else:
                        correct_output[i] += ((1 - self.search(obs, possible_handcards)) if 1 != self.env.current_player else self.search(obs, possible_handcards))

                    self.env.set_state(&state)

            self.env.set_state(&state_org)

            for i in range(32):
                model_output.p[i] = 0
            model_output.v = 0

            for i in range(correct_output.size()):
                model_output.p[self.env.players[self.env.current_player].hand_cards[i].id] = correct_output[i] / possible_handcards.size()

            self.env.regenerate_obs(&obs)
            #print([card.id for card in self.env.players[self.env.current_player].hand_cards], obs.hand_cards)
            self.model.memorize(&obs, &model_output)
        else:
            #tmp = model_output
            self.model.predict_single_p(&obs, &model_output)

           # for i in range(32):
            #    if model_output.p[i] != tmp.p[i]:
            #        print("o.o")
        return self.model.valid_step(model_output.p, &self.env.players[self.env.current_player].hand_cards)


    cpdef void solve(self, ModelRating rating):
        cdef vector[HandCards] opposite_hand_card_combinations
        cdef Observation obs

        for i in range(rating.eval_games.size()):
            for start_player in range(2):
                self.env.set_state(&rating.eval_games[i])
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

                self.env.regenerate_obs(&obs)
                self.search(obs, &opposite_hand_card_combinations)