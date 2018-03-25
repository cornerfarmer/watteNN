from libcpp.string cimport string
from libcpp.map cimport map
from libcpp.vector cimport vector
from libcpp cimport bool
from gym_watten.envs.watten_env cimport Observation, Card
from src.MCTS cimport Storage
from src cimport ModelOutput, Experience

cdef class LookUp:
    cdef map[string, Experience] table
    cdef bool watch

    cdef string generate_key(self, Observation* obs)
    cdef void memorize(self, Observation* obs, ModelOutput* value)
    cdef bool is_memorized(self, Observation* obs)
    cpdef void memorize_storage(self, Storage storage, bool clear_afterwards=?)
    cdef void predict_single(self, Observation* obs, ModelOutput* output)
    cdef int valid_step(self, float* values, vector[Card*]* hand_cards)
    cdef int argmax(self, vector[float]* values)