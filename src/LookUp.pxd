from libcpp.string cimport string
from libcpp.map cimport map
from libcpp.vector cimport vector
from libcpp cimport bool
from gym_watten.envs.watten_env cimport Observation, Card
from src.Storage cimport Storage
from src.Model cimport Model
from src cimport ModelOutput, ExperienceP, ExperienceV, StorageItem

cdef class LookUp(Model):
    cdef map[string, ExperienceP] table_p
    cdef map[string, ExperienceV] table_v
    cdef bool watch

    cdef string generate_key(self, Observation* obs)
    cdef void memorize(self, StorageItem* storage)
    cdef bool is_memorized(self, Observation* obs)
    cpdef vector[float] memorize_storage(self, Storage storage, bool clear_afterwards=?, int epochs=?, int number_of_samples=?)

    cdef void predict_p(self, vector[Observation]* obs, vector[ModelOutput]* output)
    cdef void predict_v(self, vector[Observation]* full_obs, vector[ModelOutput]* output)
    cdef void predict_single_p(self, Observation* obs, ModelOutput* output)
    cdef float predict_single_v(self, Observation* full_obs)

    cpdef void copy_weights_from(self, Model other_model)