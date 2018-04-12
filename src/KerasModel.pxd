from libcpp.string cimport string
from libcpp.map cimport map
from libcpp.vector cimport vector
from libcpp cimport bool
from gym_watten.envs.watten_env cimport Observation, Card
from src.Storage cimport Storage
from src cimport ModelOutput, Experience
from src.Model cimport Model

cdef class KerasModel(Model):
    cdef object model
    cdef object clean_opt_weights
    cdef int input_sets_size

    cpdef float memorize_storage(self, Storage storage, bool clear_afterwards=?, int epochs=?, int number_of_samples=?)
    cdef void predict_single(self, Observation* obs, ModelOutput* output)
    cpdef void copy_weights_from(self, Model other_model)
    cpdef void load(self, filename)
    cpdef void save(self, filename)