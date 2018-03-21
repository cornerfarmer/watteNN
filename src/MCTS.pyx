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

cdef struct Storage:
    vector[StorageItem] data

cdef class MCTS:

    def __cinit__(self, episodes=75, mcts_sims=20, objective_opponent=False, exploration=1):
        self.episodes = episodes
        self.mcts_sims = mcts_sims
        self.objective_opponent = objective_opponent
        self.exploration = exploration

    cdef void add_state(self, MCTSState* parent, float p, WattenEnv env, int end_v=0):
        parent.childs.push_back(MCTSState())
        parent.childs.back().n = 0
        parent.childs.back().w = 0
        parent.childs.back().v = 0
        parent.childs.back().p = p
        parent.childs.back().env_state = env.get_state()
        parent.childs.back().current_player = env.current_player
        parent.childs.back().end_v = end_v
        parent.childs.back().is_root = False


    cdef bool is_state_leaf_node(self, MCTSState* state):
        return state.childs.size() == 0


    cdef object calc_q(self, MCTSState* state, int player):
        cdef int i
        if state.end_v != 0:
            return state.end_v * (-1 if player == 1 else 1), 1
        elif state.current_player == player:
            if state.n is 0:
                return 0, 0
            else:
                return state.w / state.n, state.n
        else:
            q_sum = 0
            n_sum = 0
            for i in range(state.childs.size()):
                q_new, n = self.calc_q(&state.childs[i], player)
                q_sum += q_new * n
                n_sum += n
            return 0 if n_sum is 0 else q_sum / n_sum, n_sum


    cdef mcts_sample(self, WattenEnv env, MCTSState* state, LookUp model):
        cdef vector[Card*] hand_cards
        cdef int current_player, i
        cdef float max_u
        cdef MCTSState* max_child
        cdef ModelOutput prediction
        cdef Observation obs

        if self.is_state_leaf_node(state):
            if state.end_v != 0:
                v = state.end_v
                player = -1
            else:
                env.set_state(state.env_state)

                if model is not None:
                    obs = env.regenerate_obs()
                    prediction = model.predict_single(&obs)
                else:
                    p, v = [1] *32, [0]

                hand_cards = env.players[env.current_player].hand_cards
                current_player = env.current_player
                for card in hand_cards:
                    obs = env.step(card.id)

                    self.add_state(state, prediction.p[card.id], env, 0 if not env.is_done() else (1 if env.last_winner else -1))

                    env.set_state(state.env_state)

                v = prediction.v
                state.v = prediction.v
                player = state.current_player
        else:
            n_sum = 0
            for child in state.childs:
                n_sum += child.n

            max_child = NULL
            if not self.objective_opponent or state.current_player == 0:
                for i in range(state.childs.size()):
                    u = self.calc_q(&state.childs[i], state.current_player)[0]
                    u += self.exploration * state.childs[i].p * sqrt(n_sum) / (1 + state.childs[i].n)

                    if max_child is NULL or u > max_u:
                        max_u = u
                        max_child = &state.childs[i]
            else:
                p = [child.p for child in state.childs]
                p = np.exp(p - np.max(p))
                p /= p.sum(axis=0)
                max_child = &state.childs[np.random.choice(np.arange(0, len(p)), p=p)]

            v, player = self.mcts_sample(env, max_child, model)

        if player == state.current_player:
            state.w += v
            state.n += 1
        elif player == -1:
            state.w += v * (-1 if state.current_player == 1 else 1)
            state.n += 1
        return v, player

    cdef object mcts_game_step(self, WattenEnv env, MCTSState* root, LookUp model, int steps=0):
        if steps == 0:
            steps = self.mcts_sims

        cdef int i
        for i in range(steps):
            self.mcts_sample(env, root, model)

        if not self.objective_opponent or root.current_player == 0:
            p = []
            for i in range(root.childs.size()):
                p.append(self.calc_q(&root.childs[i], root.current_player)[0])
        else:
            p = [child.p for child in root.childs]

        p = np.exp(p - np.max(p))
        p /= p.sum(axis=0)
        return np.random.choice(np.arange(0, len(p)), p=p), p


    cdef MCTSState create_root_state(self, WattenEnv env):
        cdef MCTSState state
        state.n = 0
        state.w = 0
        state.v = 0
        state.p = 1
        state.env_state = env.get_state()
        state.current_player = env.current_player
        state.end_v = 0
        state.is_root = True
        return state

    cdef void mcts_game(self, WattenEnv env, LookUp model, Storage* storage):
        cdef Observation obs
        cdef State game_state
        cdef Card* card
        cdef int last_player, i

        obs = env.reset()
        env.current_player = random.randint(0, 1)

        cdef MCTSState root = self.create_root_state(env)
        cdef MCTSState tmp
        cdef vector[int] values

        while not env.is_done():

            storage.data.push_back(StorageItem())
            storage.data.back().obs = obs

            storage.data.back().output.v = 1 if env.current_player is 0 else -1
            values.push_back(storage.data.size() - 1)

            game_state = env.get_state()
            a, p = self.mcts_game_step(env, &root, model)
            env.set_state(game_state)

            for i in range(32):
                storage.data.back().output.p[i] = 0

            i = 0
            for card in env.players[env.current_player].hand_cards:
                storage.data.back().output.p[card.id] = p[i]
                i += 1

            last_player = env.current_player
            obs = env.step(env.players[env.current_player].hand_cards[a].id)
            tmp = root.childs[a]
            root = tmp
            root.is_root = True

        for i in values:
            storage.data[i].output.v *= (1 if env.last_winner is 0 else -1)


    cdef void mcts_generate(self, WattenEnv env, LookUp model, Storage* storage):
        cdef int i
        for i in range(self.episodes):
            self.mcts_game(env, model, storage)