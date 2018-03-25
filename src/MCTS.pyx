from gym_watten.envs.watten_env cimport State, WattenEnv, Card, Observation
from libcpp cimport bool
from libcpp.vector cimport vector
from libc.math cimport sqrt, exp
from libc.stdlib cimport rand, RAND_MAX
from src.LookUp cimport LookUp, ModelOutput
from src cimport MCTSState, StorageItem
import numpy as np
import time
import matplotlib.pyplot as plt
import pydot_ng as pydot
from io import BytesIO
cimport cython

cdef class Storage:
    pass

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


    cdef float calc_q(self, MCTSState* state, int player, int* n):
        cdef int i, n_sum, n_child
        cdef float q_sum
        if state.end_v != 0:
            n[0] = 1
            return state.end_v * (-1 if player == 1 else 1)
        elif state.current_player == player:
            if state.n == 0:
                n[0] = 0
                return 0
            else:
                n[0] = state.n
                return state.w / state.n
        else:
            q_sum = 0
            n_sum = 0

            for i in range(state.childs.size()):
                q_sum += self.calc_q(&state.childs[i], player, &n_child) * n_child
                n_sum += n_child

            n[0] = n_sum
            return 0 if n_sum is 0 else q_sum / n_sum

    @cython.cdivision(True)
    cdef float mcts_sample(self, WattenEnv env, MCTSState* state, LookUp model, int* player):
        cdef int current_player, i
        cdef float max_u
        cdef float u, v
        cdef MCTSState* max_child
        cdef int child_n, n_sum
        if self.is_state_leaf_node(state):
            if state.end_v != 0:
                v = state.end_v
                player[0] = -1
            else:
                env.set_state(&state.env_state)

                if model is not None:
                    env.regenerate_obs(&self._obs)
                    model.predict_single(&self._obs, &self._prediction)
                #else:
                #    p, v = [1] *32, 0

                self._hand_cards = env.players[env.current_player].hand_cards
                current_player = env.current_player

                for card in self._hand_cards:
                    env.step(card.id, &self._obs)
                    self.add_state(state, self._prediction.p[card.id], env, 0 if not env.is_done() else (1 if env.last_winner == 0 else -1))
                    env.set_state(&state.env_state)

                v = self._prediction.v
                state.v = self._prediction.v
                player[0] = state.current_player

        else:

            n_sum = 0
            for i in range(state.childs.size()):
                n_sum += state.childs[i].n

            max_child = NULL
            if not self.objective_opponent or state.current_player == 0:
                for i in range(state.childs.size()):
                    u = self.calc_q(&state.childs[i], state.current_player, &child_n)
                    u += self.exploration * state.childs[i].p * sqrt(n_sum) / (1 + state.childs[i].n)

                    if max_child is NULL or u > max_u:
                        max_u = u
                        max_child = &state.childs[i]
            #else:
            #    p = [child.p for child in state.childs]
            #    p = np.exp(p - np.max(p))
            #    p /= p.sum(axis=0)
            #    max_child = &state.childs[np.random.choice(np.arange(0, len(p)), p=p)]


            v = self.mcts_sample(env, max_child, model, player)

        if player[0] == state.current_player:
            state.w += v
            state.n += 1
        elif player[0] == -1:
            state.w += v * (-1 if state.current_player == 1 else 1)
            state.n += 1
        return v

    cdef int mcts_game_step(self, WattenEnv env, MCTSState* root, LookUp model, vector[float]* p, int steps=0):
        if steps == 0:
            steps = self.mcts_sims

        cdef int i, player
        for i in range(steps):
            self.mcts_sample(env, root, model, &player)

        cdef int child_n
        p.clear()
        if not self.objective_opponent or root.current_player == 0:
            for i in range(root.childs.size()):
                p.push_back(self.calc_q(&root.childs[i], root.current_player, &child_n))
        else:
            for i in range(root.childs.size()):
                p.push_back(root.childs[i].p)

        cdef float pmax = -1
        for i in range(p.size()):
            if pmax < p[0][i]:
                pmax = p[0][i]

        cdef float psum = 0
        for i in range(p.size()):
            p[0][i] = exp(p[0][i] - pmax)
            psum += p[0][i]

        for i in range(p.size()):
            p[0][i] /= psum

        cdef float r = <float>rand() / RAND_MAX

        for i in range(p.size()):
            r -= p[0][i]
            if r <= 0:
                return i
        return p.size() - 1


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

    cdef void mcts_game(self, WattenEnv env, LookUp model, Storage storage):
        cdef Observation obs
        cdef State game_state
        cdef Card* card
        cdef int last_player, i

        env.reset(&obs)
        if self.objective_opponent:
            env.current_player = rand() % 2

        cdef MCTSState root = self.create_root_state(env)
        cdef MCTSState tmp
        cdef vector[int] values
        cdef vector[float] p

        while not env.is_done():

            storage.data.push_back(StorageItem())
            storage.data.back().obs = obs

            storage.data.back().output.v = 1 if env.current_player is 0 else -1
            values.push_back(storage.data.size() - 1)

            game_state = env.get_state()
            a = self.mcts_game_step(env, &root, model, &p)
            env.set_state(&game_state)

            for i in range(32):
                storage.data.back().output.p[i] = 0

            i = 0
            for card in env.players[env.current_player].hand_cards:
                storage.data.back().output.p[card.id] = p[i]
                i += 1

            last_player = env.current_player
            env.step(env.players[env.current_player].hand_cards[a].id, &obs)
            tmp = root.childs[a]
            root = tmp
            root.is_root = True

        for i in values:
            storage.data[i].output.v *= (1 if env.last_winner is 0 else -1)


    cpdef void mcts_generate(self, WattenEnv env, LookUp model, Storage storage):
        cdef int i
        for i in range(self.episodes):
            self.mcts_game(env, model, storage)


    cdef void draw_tree(self, MCTSState* root, int tree_depth=5, object tree_path=[]):
        dot = pydot.Dot()
        dot.set('rankdir', 'TB')
        dot.set('concentrate', True)
        dot.set_node_defaults(shape='record')

        self.create_nodes(root, dot, tree_depth, tree_path)

       # print("Root: " + str(root.end_v if root.n is 0 else root.w / root.n) + " / " + str(root.n))
       # for child in root.childs:
       #     print( str(child.end_v if child.n is 0 else child.w / child.n) + " / " + str(child.n) + " p: " + str(child.p))

        # render pydot by calling dot, no file saved to disk
        png_str = dot.create_png(prog='dot')
        dot.write_svg('tree.svg')

        # treat the dot output string as an image file
        sio = BytesIO()
        sio.write(png_str)
        sio.seek(0)

        # plot the image
        fig, ax = plt.subplots(figsize=(18, 5))
        ax.imshow(plt.imread(sio), interpolation="bilinear")

    cdef object create_nodes(self, MCTSState* root, object dot, int tree_depth, object tree_path, int id=0):
        text = "N: " + str(root.n) + " (" + str(root.current_player) + ')\n'
        text += "Q: " + str(root.end_v if root.end_v is not 0 or root.n is 0 else root.w / root.n) + '\n'
        text += "P: " + str(root.p) + '\n'
        text += "V: " + str(root.v)

        node = pydot.Node(str(id), label=text)
        dot.add_node(node)
        id += 1

        if tree_depth > 1:
            i = 0
            for i in range(root.childs.size()):
                if len(tree_path) == 0 or (type(tree_path[0]) == list and i in tree_path[0] or type(tree_path[0]) == int and tree_path[0] == i):
                    child_node, id = self.create_nodes(&root.childs[i], dot, tree_depth - 1, tree_path if len(tree_path) == 0 else tree_path[1:], id)
                    dot.add_edge(pydot.Edge(node.get_name(), child_node.get_name()))
                i+=1

        return node, id

    def end(self):
        pass