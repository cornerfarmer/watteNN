from libcpp.string cimport string
from libcpp.map cimport map
from libcpp.vector cimport vector
from libcpp cimport bool
from gym_watten.envs.watten_env cimport Observation, Card
from src.Storage cimport Storage
from src cimport ModelOutput

cdef class Model:
    cpdef vector[float] memorize_storage(self, Storage storage, bool clear_afterwards=?, int epochs=?, int number_of_samples=?)
    cdef void predict_p(self, vector[Observation]* obs, vector[ModelOutput]* output)
    cdef void predict_v(self, vector[Observation]* full_obs, vector[ModelOutput]* output)
    cdef void predict(self, vector[Observation]* full_obs, vector[Observation]* obs, vector[ModelOutput]* output)
    cdef void predict_single_p(self, Observation* obs, ModelOutput* output)
    cdef float predict_single_v(self, Observation* full_obs)
    cdef void predict_single(self, Observation* full_obs, Observation* obs, ModelOutput* output)
    cdef int valid_step(self, float* values, vector[Card*]* hand_cards)
    cdef int argmax(self, vector[float]* values)
    cpdef void copy_weights_from(self, Model other_model)
    cpdef void load(self, filename)
    cpdef void save(self, filename)