from gym_watten.envs.watten_env cimport State, WattenEnv, Card, Observation
from src.LookUp cimport LookUp, Model, ModelOutput
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.map cimport map

ctypedef map[int, float] CacheEntry
ctypedef vector[Card*] HandCards

cdef class ModelRating:
    cdef WattenEnv env
    cdef vector[State] eval_games
    cdef LookUp memory

    cdef void _define_eval_games(self)
    cdef vector[float] calc_correct_output_sample(self, State* state, Model model, vector[HandCards]* possible_hand_cards)
    cdef int calc_correct_output(self, State state, Model model, vector[HandCards]* possible_hand_cards)
    cdef int calc_exploitability_in_game(self, Model model, vector[HandCards]* possible_hand_cards)
    cpdef float calc_exploitability(self, Model model)