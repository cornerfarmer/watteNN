from libcpp.string cimport string
from libcpp.map cimport map
from libcpp.vector cimport vector
from libcpp cimport bool
from gym_watten.envs.watten_env cimport Observation, Card, WattenEnv
from src.Storage cimport Storage
from src cimport ModelOutput, Experience
from src.Model cimport Model

cdef class KerasModel(Model):
    cdef object play_model, choose_model
    cdef object clean_opt_weights
    cdef int play_input_sets_size
    cdef int choose_input_sets_size
    cdef int batch_size
    cdef float lr
    cdef float momentum

    cdef void _build_choose_model(self, WattenEnv env, int hidden_neurons)
    cdef void _build_play_model(self, WattenEnv env, int hidden_neurons)
    cpdef vector[float] memorize_storage(self, Storage storage, bool clear_afterwards=?, int epochs=?, int number_of_samples=?)
    cdef void predict_single(self, Observation* obs, ModelOutput* output)
    cpdef void copy_weights_from(self, Model other_model)
    cpdef void load(self, filename)
    cpdef void save(self, filename)