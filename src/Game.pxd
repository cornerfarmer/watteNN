from gym_watten.envs.watten_env cimport WattenEnv, Observation, State
from src.LookUp cimport LookUp
from libcpp.vector cimport vector

cdef class Game:
    cdef WattenEnv env

    cdef int match(self, LookUp agent1, LookUp agent2, render=?, reset=?)
    cdef float compare_given_games(self, LookUp agent1, LookUp agent2, vector[State]* games)