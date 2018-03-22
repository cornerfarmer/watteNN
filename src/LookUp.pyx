from libcpp.string cimport string
from gym_watten.envs.watten_env cimport Observation, WattenEnv

cdef struct ModelOutput:
    float p[32]
    float v

cdef struct Experience:
    ModelOutput output
    int n

cdef extern from "<string>" namespace "std":
    string to_string(int val)

cdef class LookUp:

    cdef string generate_key(self, Observation* obs):
        cdef int i, j
        cdef string card_ids, table_card

        for i in range(4):
            for j in range(8):
                if obs.hand_cards[i][j][0] > 0:
                    card_ids += to_string(j + i * 8) + <char*>","
                if obs.hand_cards[i][j][1] > 0:
                    table_card = to_string(j + i * 8)

        return card_ids + <char*>"-" + table_card + <char*>"-" + to_string(obs.tricks[0] * 1 + obs.tricks[1] * 2) + <char*>"-" + to_string(obs.tricks[2] * 1 + obs.tricks[3] * 2)


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

    cdef void predict_single(self, Observation* obs, ModelOutput* output):
        key = self.generate_key(obs)

        cdef int i
        if self.table.count(key) > 0:
            for i in range(32):
                output.p[i] = self.table[key].output.p[i] / self.table[key].n
            output.v = self.table[key].output.v / self.table[key].n
        else:
            for i in range(32):
                output.p[i] = 1
            output.v = 0