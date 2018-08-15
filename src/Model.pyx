from libcpp.string cimport string
from libcpp cimport bool
from gym_watten.envs.watten_env cimport Observation, WattenEnv, Card
from src.MCTS cimport Storage
from src cimport ModelOutput


cdef extern from "<string>" namespace "std":
    string to_string(int val)

cdef class Model:

    cpdef vector[float] memorize_storage(self, Storage storage, bool clear_afterwards=True, int epochs=1, int number_of_samples=0):
        raise NotImplementedError('subclasses must override memorize_storage()!')

    cdef void predict_single_p(self, Observation* obs, ModelOutput* output):
        raise NotImplementedError('subclasses must override predict_single_p()!')

    cdef float predict_single_v(self, Observation* full_obs):
        raise NotImplementedError('subclasses must override predict_single_v()!')

    cdef void predict_single(self, Observation* full_obs, Observation* obs, ModelOutput* output):
        self.predict_single_p(obs, output)
        output.v = self.predict_single_v(full_obs)

    cdef int valid_step(self, float* values, vector[Card*]* hand_cards):
        cdef float max_value
        cdef Card* card, *max_card = NULL

        for card in hand_cards[0]:
            if max_card is NULL or max_value < values[card.id]:
                max_card = card
                max_value = values[card.id]

        return max_card.id

    cdef int argmax(self, vector[float]* values):
        cdef float max_value
        cdef int max_index = -1
        for i in range(0, values.size()):
            if max_index == -1 or values[0][i] > max_value:
                max_index = i
                max_value = values[0][i]
        return max_index

    cpdef void copy_weights_from(self, Model other_model):
        raise NotImplementedError('subclasses must override copy_weights_from()!')

    cpdef void load(self, filename):
        raise NotImplementedError('subclasses must override load()!')

    cpdef void save(self, filename):
        raise NotImplementedError('subclasses must override load()!')