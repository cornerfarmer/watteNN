from libcpp.vector cimport vector
from src cimport MCTSState, StorageItem

cdef class Storage:
    cdef vector[StorageItem] data
    cdef int max_samples
    cdef int next_index
    cdef int number_of_samples

    cdef int add_item(self)

