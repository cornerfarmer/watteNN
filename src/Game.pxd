from gym_watten.envs.watten_env cimport WattenEnv, Observation, State
from src.LookUp cimport LookUp
from src.ModelRating cimport ModelRating
from libcpp cimport bool
from libcpp.vector cimport vector

cdef class Game:
    cdef WattenEnv env

    cpdef int match(self, LookUp agent1, LookUp agent2, bool render=?, bool reset=?)
    cpdef float compare_given_games(self, LookUp agent1, LookUp agent2, ModelRating rating)