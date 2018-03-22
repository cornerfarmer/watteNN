from gym_watten.envs.watten_env cimport State, WattenEnv, Card, Observation
from libcpp cimport bool
from libcpp.vector cimport vector
from libc.math cimport sqrt
from src.LookUp cimport LookUp, ModelOutput
import numpy as np
import random

cdef struct MCTSState:
    vector[MCTSState] childs
    int n
    float w
    float v
    float p
    State env_state
    float end_v
    int current_player
    bool is_root

cdef struct StorageItem:
    Observation obs
    ModelOutput output

cdef class Storage:
    cdef vector[StorageItem] data

cdef class MCTS:

    cdef int episodes
    cdef int mcts_sims
    cdef bool objective_opponent
    cdef float exploration

    cdef void add_state(self, MCTSState* parent, float p, WattenEnv env, int end_v=?)
    cdef bool is_state_leaf_node(self, MCTSState* state)
    cdef float calc_q(self, MCTSState* state, int player, int* n)
    cdef float mcts_sample(self, WattenEnv env, MCTSState* state, LookUp model, int* player)
    cdef int mcts_game_step(self, WattenEnv env, MCTSState* root, LookUp model, vector[float]* p, int steps=?)
    cdef MCTSState create_root_state(self, WattenEnv env)
    cdef void mcts_game(self, WattenEnv env, LookUp model, Storage storage)
    cdef void mcts_generate(self, WattenEnv env, LookUp model, Storage storage)
    cdef void draw_tree(self, MCTSState* root, int tree_depth=?, object tree_path=?)
    cdef object create_nodes(self, MCTSState* root, object dot, int tree_depth, object tree_path, int id=?)