from libcpp.vector cimport vector
from libcpp cimport bool
from src cimport MCTSState, StorageItem
from gym_watten.envs.watten_env cimport WattenEnv

cdef class Storage:
    cdef vector[StorageItem] data
    cdef int max_samples
    cdef int next_index
    cdef int number_of_samples

    cdef int add_item(self)
    cpdef void export_csv(self, file_name, WattenEnv env)

