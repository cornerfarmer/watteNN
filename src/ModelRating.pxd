from gym_watten.envs.watten_env cimport State, WattenEnv, Card, Observation
from src.LookUp cimport LookUp, ModelOutput
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.map cimport map

ctypedef map[int, float] CacheEntry
ctypedef vector[Card*] HandCards

cdef class ModelRating:
    cdef WattenEnv env
    cdef vector[State] eval_games
    cdef map[string, CacheEntry] cache

    cdef void _define_eval_games(self)
    cdef int search(self, Observation obs, LookUp model, vector[float]* all_values=?, target_player=?)
    cdef string generate_hand_cards_key(self, vector[Card*]* hand_cards)
    cdef string generate_cache_key(self, State* state)
    cdef vector[float] calc_correct_output_sample(self, State* state, LookUp model)
    cdef vector[float] calc_correct_output(self, State state, LookUp model, vector[HandCards]* possible_hand_cards)
    cdef int calc_exploitability_in_game(self, LookUp model, vector[HandCards]* possible_hand_cards)
    cpdef float calc_exploitability(self, model)