from gym_watten.envs.watten_env cimport WattenEnv, Observation, State
from src.Model cimport Model
from src.ModelRating cimport ModelRating
from libcpp cimport bool
from libcpp.vector cimport vector



cdef class Game:
    cdef WattenEnv env
    cdef public float mean_game_length
    cdef public float mean_v_p1
    cdef public float mean_v_p2
    cdef float v_loss_on_sum
    cdef float v_loss_off_sum
    cdef int v_loss_on_n
    cdef int v_loss_off_n

    cpdef int match(self, Model agent1, Model agent2, bool render=?, bool reset=?)
    cpdef float compare_given_games(self, Model agent1, Model agent2, ModelRating rating)
    cpdef float compare_rand_games(self, Model agent1, Model agent2, int number_of_games)
    cdef game_tree_step(self, Model model, Observation obs, dot, parent_node, prob, joint_prob, next_id, key, table, tree_only, true_edge_prob, exploration_mode)
    cpdef draw_game_tree(self, Model model, ModelRating modelRating, use_cache, tree_ind, debug_tree_key, tree_path=?)