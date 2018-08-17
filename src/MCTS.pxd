from gym_watten.envs.watten_env cimport State, WattenEnv, Card, Observation
from libcpp cimport bool
from libcpp.vector cimport vector
from libc.math cimport sqrt
from src.Model cimport Model
from src.ModelRating cimport ModelRating
from src cimport ModelOutput
from src cimport MCTSState, StorageItem
from src.Storage cimport Storage

cdef class MCTS:

    cdef int episodes
    cdef int mcts_sims
    cdef bool objective_opponent
    cdef float exploration
    cdef bool high_q_for_unvisited_nodes

    cdef vector[Card*] _hand_cards
    cdef ModelOutput _prediction
    cdef Observation _obs
    cdef Observation _full_obs
    cdef object timing

    cdef void add_state(self, MCTSState* parent, float p, WattenEnv env, int end_v=?)
    cdef bool is_state_leaf_node(self, MCTSState* state)
    cdef float calc_q(self, MCTSState* state, int player, int* n)
    cdef float mcts_sample(self, WattenEnv env, MCTSState* state, Model model)
    cdef int softmax_step(self, vector[float]* p, bool do_exploration=?)
    cdef int mcts_game_step(self, WattenEnv env, MCTSState* root, Model model, vector[float]* p, int steps=?)
    cdef MCTSState create_root_state(self, WattenEnv env)
    cdef void mcts_game(self, WattenEnv env, Model model, Storage storage, bool new_game=?)
    cpdef void mcts_generate(self, WattenEnv env, Model model, Storage storage)
    cdef void draw_tree(self, MCTSState* root, int tree_depth=?, object tree_path=?)
    cdef object create_nodes(self, MCTSState* root, object dot, int tree_depth, object tree_path, int id=?)
    cpdef draw_game_tree(self, ModelRating rating, WattenEnv env, Model model, int game_index, int tree_depth, pre_actions)
    cpdef create_storage(self, ModelRating rating, WattenEnv env, Model model, Storage storage, int game_index)