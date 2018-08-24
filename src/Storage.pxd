from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool
from src cimport MCTSState, StorageItem
from gym_watten.envs.watten_env cimport WattenEnv

cdef class Storage:
    cdef vector[StorageItem] data
    cdef int max_samples
    cdef int next_index
    cdef int number_of_samples
    cdef object key_numbers

    cdef int add_item(self, string key)
    cpdef void export_csv(self, file_name, WattenEnv env)
    cdef void clear(self)

