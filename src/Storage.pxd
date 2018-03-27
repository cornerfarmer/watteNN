from libcpp.vector cimport vector
from src cimport MCTSState, StorageItem

cdef class Storage:
    cdef vector[StorageItem] data
