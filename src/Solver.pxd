from gym_watten.envs.watten_env cimport State, WattenEnv, Card, Observation
from src.LookUp cimport LookUp, ModelOutput
from src.ModelRating cimport ModelRating, HandCards
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.map cimport map
from libcpp.algorithm cimport sort
import itertools
from libc.string cimport memset
import sys

cdef class Solver:
    cdef LookUp model
    cdef WattenEnv env

    cdef float search(self, Observation obs, vector[HandCards]* possible_handcards)
    cdef int choose_step(self, Observation obs, vector[HandCards]* possible_handcards)
    cpdef void solve(self, ModelRating rating)