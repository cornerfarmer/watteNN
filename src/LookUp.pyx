from libcpp.string cimport string
from libcpp cimport bool
from gym_watten.envs.watten_env cimport Observation, WattenEnv, Card
from src.MCTS cimport Storage
from src.Model cimport Model
from src cimport ModelOutput


cdef extern from "<string>" namespace "std":
    string to_string(int val)

cdef class LookUp(Model):
    def __cinit__(self):
        self.watch = False

    cdef string generate_key(self, Observation* obs):
        cdef int i, j
        cdef string card_ids, table_card, trick_card_0, trick_card_1, trick_card_2, trick_card_3

        for i in range(4):
            for j in range(8):
                if obs.hand_cards[i][j][0] > 0:
                    card_ids += to_string(j + i * 8) + <char*>","
                if obs.hand_cards[i][j][1] > 0:
                    table_card = to_string(j + i * 8)
                if obs.hand_cards[i][j][2] > 0:
                    trick_card_0 = to_string(j + i * 8)
                if obs.hand_cards[i][j][3] > 0:
                    trick_card_1 = to_string(j + i * 8)
                if obs.hand_cards[i][j][4] > 0:
                    trick_card_2 = to_string(j + i * 8)
                if obs.hand_cards[i][j][5] > 0:
                    trick_card_3 = to_string(j + i * 8)

        return card_ids + <char*>"-" + table_card + <char*>"-" + trick_card_0 + <char*>"-" + trick_card_1 + <char*>"-" + trick_card_2 + <char*>"-" + trick_card_3 + <char*>"-" + to_string(obs.tricks[0] * 1 + obs.tricks[1] * 2) + <char*>"-" + to_string(obs.tricks[2] * 1 + obs.tricks[3] * 2)


    cdef void memorize(self, Observation* obs, ModelOutput* value):
        cdef string key = self.generate_key(obs)
        cdef int i

        if self.table.count(key) > 0:
            for i in range(32):
                self.table[key].output.p[i] += value.p[i]

            self.table[key].output.v += value.v
            self.table[key].n += 1
        else:
            self.table[key] = Experience()
            self.table[key].output.p = value.p
            self.table[key].output.v = value.v
            self.table[key].n = 1

    cdef bool is_memorized(self, Observation* obs):
        return self.table.count(self.generate_key(obs)) > 0

    cpdef float memorize_storage(self, Storage storage, bool clear_afterwards=True, int epochs=1, int number_of_samples=0):
        cdef int i
        for i in range(storage.data.size()):
            self.memorize(&storage.data[i].obs, &storage.data[i].output)

        if clear_afterwards:
            storage.data.clear()
        return 0

    cdef void predict_single(self, Observation* obs, ModelOutput* output):
        key = self.generate_key(obs)

        cdef int i
        if self.table.count(key) > 0:
            for i in range(32):
                output.p[i] = self.table[key].output.p[i] / self.table[key].n
            output.v = self.table[key].output.v / self.table[key].n
        else:
            if self.watch:
                print(key)
            for i in range(32):
                output.p[i] = 1
            output.v = 0

    cpdef void copy_weights_from(self, Model other_model):
        self.table = (<LookUp>other_model).table