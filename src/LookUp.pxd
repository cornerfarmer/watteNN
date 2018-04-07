from libcpp.string cimport string
from libcpp.map cimport map
from libcpp.vector cimport vector
from libcpp cimport bool
from gym_watten.envs.watten_env cimport Observation, Card
from src.Storage cimport Storage
from src.Model cimport Model
from src cimport ModelOutput, Experience

cdef class LookUp(Model):
    cdef map[string, Experience] table
    cdef bool watch

    cdef string generate_key(self, Observation* obs)
    cdef void memorize(self, Observation* obs, ModelOutput* value)
    cdef bool is_memorized(self, Observation* obs)
    cpdef float memorize_storage(self, Storage storage, bool clear_afterwards=?, int epochs=?, int number_of_samples=?)
    cdef void predict_single(self, Observation* obs, ModelOutput* output)
    cpdef void copy_weights_from(self, Model other_model)