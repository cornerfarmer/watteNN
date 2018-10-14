from libcpp.string cimport string
from libcpp.map cimport map
from libcpp.vector cimport vector
from libcpp cimport bool
from gym_watten.envs.watten_env cimport Observation, Card, WattenEnv
from src.Storage cimport Storage
from src cimport ModelOutput
from src.Model cimport Model

cdef class KerasModel(Model):
    cdef object play_model, choose_model, value_model
    cdef object clean_opt_weights
    cdef int play_input_sets_size
    cdef int play_input_scalars_size
    cdef int choose_input_sets_size
    cdef int batch_size
    cdef float policy_lr
    cdef float policy_momentum
    cdef float value_lr
    cdef float value_momentum
    cdef float clip
    cdef float equalizer
    cdef Observation test_obs
    cdef Observation test_obs_p

    cdef void _build_choose_model(self, WattenEnv env, int hidden_neurons)
    cdef void _build_play_model(self, WattenEnv env, int hidden_neurons)
    cdef void _build_value_model(self, WattenEnv env, int hidden_neurons)
    cpdef vector[float] memorize_storage(self, Storage storage, bool clear_afterwards=?, int epochs=?, int number_of_samples=?)
    cdef void predict_p(self, vector[Observation]* obs, vector[ModelOutput]* output)
    cdef void predict_v(self, vector[Observation]* full_obs, vector[ModelOutput]* output)
    cdef object _clip_output(self, output)
    cdef void predict_single_p(self, Observation* obs, ModelOutput* output)
    cdef float predict_single_v(self, Observation* full_obs)
    cpdef void copy_weights_from(self, Model other_model)
    cpdef void load(self, filename)
    cpdef void save(self, filename)
    cpdef predict_v_model(self, epoch, logs)
    cpdef predict_p_model(self, epoch, logs)