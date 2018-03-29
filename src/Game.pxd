from gym_watten.envs.watten_env cimport WattenEnv, Observation, State
from src.Model cimport Model
from src.ModelRating cimport ModelRating
from libcpp cimport bool
from libcpp.vector cimport vector

cdef class Game:
    cdef WattenEnv env

    cpdef int match(self, Model agent1, Model agent2, bool render=?, bool reset=?)
    cpdef float compare_given_games(self, Model agent1, Model agent2, ModelRating rating)