from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool
from gym_watten.envs.watten_env cimport Observation, WattenEnv, Card
from src.MCTS cimport Storage
from src.Model cimport Model
from src cimport ModelOutput, ExperienceP, ExperienceV, StorageItem
import pickle

cdef extern from "<string>" namespace "std":
    string to_string(int val)

cdef class LookUp(Model):
    def __cinit__(self):
        self.watch = False

    cdef string generate_key(self, Observation* obs):
        cdef int i, j
        cdef string trick_ids, hand_cards

        for k in range(obs.sets[0][0].size()):
            for i in range(4):
                for j in range(8):
                    if obs.sets[i][j][k] > 0:
                        if k >= 2:
                            trick_ids = string(<char*>",") + to_string(j + i * 8) + <char*>"." + to_string(obs.scalars[k - 2 + 4]) + trick_ids
                        elif k == 1:
                            trick_ids = string(<char*>",") + to_string(j + i * 8)  + <char*>".0" + trick_ids
                        else:
                            hand_cards += to_string(j + i * 8) + <char*>","

        return trick_ids + <char*>"-" + hand_cards


    cdef void memorize(self, StorageItem* storage):
        cdef string key = self.generate_key(&storage.obs)
        cdef int target_index
        cdef int sum_p = 0

        for i in range(32):
            sum_p += storage.obs.sets[i / 8][i % 8][0]

        if self.table_p.count(key) == 0:
            self.table_p[key] = ExperienceP()
            for i in range(32):
                self.table_p[key].p[i] = storage.obs.sets[i / 8][i % 8][0] / <float>(sum_p)

        cdef float prev, added = 0
        for i in range(32):
            if storage.obs.sets[i / 8][i % 8][0] == 1 and storage.output.p[i] == 1:
                prev = self.table_p[key].p[i]
                self.table_p[key].p[i] += storage.weight * 0.005
                self.table_p[key].p[i] = min(self.table_p[key].p[i], 1)
                added = self.table_p[key].p[i] - prev

        cdef int left = sum_p - 1
        cdef float per_action
        while added > 1e-6:
            left = 0
            for i in range(32):
                if storage.obs.sets[i / 8][i % 8][0] == 1 and storage.output.p[i] != 1 and self.table_p[key].p[i] > 0:
                    left += 1
            if left == 0:
                break

            per_action = added / left

            for i in range(32):
                if storage.obs.sets[i / 8][i % 8][0] == 1 and storage.output.p[i] != 1:
                    prev = self.table_p[key].p[i]

                    self.table_p[key].p[i] -= per_action
                    if self.table_p[key].p[i] < 0:
                        self.table_p[key].p[i] = 0

                    added -= prev - self.table_p[key].p[i]



    cdef bool is_memorized(self, Observation* obs):
        return self.table_p.count(self.generate_key(obs)) > 0

    cpdef vector[float] memorize_storage(self, Storage storage, bool clear_afterwards=True, int epochs=1, int number_of_samples=0):
        cdef int i
        for i in range(storage.number_of_samples):
            if not storage.data[i].value_net:
                self.memorize(&storage.data[i])

        if clear_afterwards:
            storage.clear()

        cdef vector[float] loss
        loss.push_back(0)
        loss.push_back(0)
        loss.push_back(0)
        return loss

    cdef void predict_single_p(self, Observation* obs, ModelOutput* output):
        key = self.generate_key(obs)

        cdef int i, number_of_cards = 0
        if self.table_p.count(key) > 0:
            for i in range(32):
                if self.table_p[key].p[i] < 0.02:
                    output.p[i] = 0
                elif self.table_p[key].p[i] > 0.98:
                    output.p[i] = 1
                else:
                    output.p[i] = self.table_p[key].p[i]

            output.v = 0
        else:

            for i in range(32):
                if obs.sets[i / 8][i % 8][0] == 1:
                    number_of_cards += 1

            for i in range(32):
                if obs.sets[i / 8][i % 8][0] == 1:
                    output.p[i] = 1.0 / number_of_cards
            output.v = 0

    cdef float predict_single_v(self, Observation* full_obs):
        return 0


    cdef void predict_p(self, vector[Observation]* obs, vector[ModelOutput]* output):
        cdef int i

        for i in range(output.size()):
            self.predict_single_p(&obs[0][i], &output[0][i])

    cdef void predict_v(self, vector[Observation]* full_obs, vector[ModelOutput]* output):
        for i in range(full_obs.size()):
            output[0][i].v = self.predict_single_v(&full_obs[0][i])


    cpdef void copy_weights_from(self, Model other_model):
        self.table_p = (<LookUp>other_model).table_p

    cpdef void load(self, filename):
        with open(str(filename + "model.pk"), 'rb') as handle:
            pydict = pickle.load(handle)

        self.table_p = pydict

    cpdef void save(self, filename):
        pydict = self.table_p

        with open(str(filename + "model.pk"), 'wb') as handle:
            pickle.dump(pydict, handle)