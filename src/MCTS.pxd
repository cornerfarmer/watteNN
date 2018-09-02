from gym_watten.envs.watten_env cimport State, WattenEnv, Card, Observation
from libcpp cimport bool
from libcpp.vector cimport vector
from libc.math cimport sqrt
from src.Model cimport Model
from src.ModelRating cimport ModelRating
from src cimport ModelOutput
from src cimport MCTSState, StorageItem
from src.Storage cimport Storage

cdef class PredictionQueue:
    cdef vector[Observation] obs_queue
    cdef vector[Observation] full_obs_queue
    cdef vector[ModelOutput] outputs
    cdef vector[int] worker_ids

    cdef enqueue(self, Observation* full_obs, Observation* obs, int worker_id)
    cdef do_prediction(self, WattenEnv env, Model model, object worker)

cdef class MCTSWorker:
    cdef Storage value_storage
    cdef int current_step
    cdef MCTSState* prediction_state
    cdef MCTSState* root
    cdef vector[MCTSState*] nodes
    cdef bool finished
    cdef int worker_id
    cdef int mcts_sims
    cdef float exploration
    cdef bool only_one_step
    cdef bool exploration_mode
    cdef int exploration_player
    cdef float step_exploration

    cdef Observation _obs
    cdef Observation _full_obs

    cdef void _clear_nodes(self)
    cdef void reset(self, env)
    cdef void add_state(self, MCTSState* parent, float p, WattenEnv env, int end_v=?, float scale=?)
    cdef bool is_state_leaf_node(self, MCTSState* state)
    cdef void handle_leaf_state(self, MCTSState* state, float v)
    cdef void handle_prediction(self, WattenEnv env, ModelOutput* prediction)
    cdef bool mcts_sample(self, WattenEnv env, MCTSState* state, PredictionQueue queue)
    cdef int softmax_step(self, vector[float]* p)
    cdef bool mcts_game_step(self, WattenEnv env, PredictionQueue queue, vector[float]* p, float* scale, int* action, bool* exploration_mode_activated)
    cdef MCTSState create_root_state(self, WattenEnv env)
    cdef bool mcts_game(self, WattenEnv env, PredictionQueue queue, Storage storage)

cdef class MCTS:

    cdef int episodes

    cdef vector[Card*] _hand_cards
    cdef ModelOutput _prediction
    cdef object worker

    cpdef void mcts_generate(self, WattenEnv env, Model model, Storage storage, bool reset_env=?)
    cdef void draw_tree(self, MCTSState* root, int tree_depth=?, object tree_path=?)
    cdef object create_nodes(self, MCTSState* root, object dot, int tree_depth, object tree_path, int id=?)
    cpdef draw_game_tree(self, ModelRating rating, WattenEnv env, Model model, Storage storage, int game_index, int tree_depth, pre_actions)
    cpdef create_storage(self, ModelRating rating, WattenEnv env, Model model, Storage storage, int game_index)