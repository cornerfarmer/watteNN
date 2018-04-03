from libcpp.string cimport string
from libcpp.map cimport map
from libcpp.vector cimport vector
from libcpp cimport bool
from gym_watten.envs.watten_env cimport Observation, Card
from src.Storage cimport Storage
from src cimport ModelOutput, Experience

cdef class Model:
    cpdef void memorize_storage(self, Storage storage, bool clear_afterwards=?, int epochs=?, int number_of_samples=?)
    cdef void predict_single(self, Observation* obs, ModelOutput* output)
    cdef int valid_step(self, float* values, vector[Card*]* hand_cards)
    cdef int argmax(self, vector[float]* values)
    cpdef void copy_weights_from(self, Model other_model)