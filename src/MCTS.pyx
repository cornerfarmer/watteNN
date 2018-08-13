from gym_watten.envs.watten_env cimport State, WattenEnv, Card, Observation
from libcpp cimport bool
from libcpp.vector cimport vector
from libc.math cimport sqrt, exp
from libc.stdlib cimport rand, RAND_MAX
from src.Model cimport Model, ModelOutput
from src.ModelRating cimport ModelRating
from src cimport MCTSState, StorageItem
import numpy as np
import time
import matplotlib.pyplot as plt
import pydot_ng as pydot
from io import BytesIO
cimport cython
from src.Storage cimport Storage

cdef class MCTS:

    def __cinit__(self, episodes=75, mcts_sims=20, objective_opponent=False, exploration=1, high_q_for_unvisited_nodes=False):
        self.episodes = episodes
        self.mcts_sims = mcts_sims
        self.objective_opponent = objective_opponent
        self.exploration = exploration
        self.high_q_for_unvisited_nodes = high_q_for_unvisited_nodes

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
                if self.high_q_for_unvisited_nodes:
                    n[0] = 1
                    return 1
                else:
                    n[0] = 0
                    return -1
            else:
                n[0] = state.n
                return state.w / state.n
        else:
            if state.childs.size() == 0 and self.high_q_for_unvisited_nodes:
                n[0] = 1
                return 1

            q_sum = 0
            n_sum = 0

            for i in range(state.childs.size()):
                q_sum += self.calc_q(&state.childs[i], player, &n_child) * n_child
                n_sum += n_child

            n[0] = n_sum
            return 0 if n_sum is 0 else q_sum / n_sum

    @cython.cdivision(True)
    cdef float mcts_sample(self, WattenEnv env, MCTSState* state, Model model, int* player):
        cdef int current_player, i
        cdef float max_u
        cdef float u, v
        cdef MCTSState* max_child
        cdef int child_n, n_sum
        cdef vector[float] p
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

            for i in range(state.childs.size()):
                p.push_back(state.childs[i].p)
            max_child = &state.childs[self.softmax_step(&p)]

            v = self.mcts_sample(env, max_child, model, player)

        if player[0] == state.current_player:
            state.w += v
            state.n += 1
        elif player[0] == -1:
            state.w += v * (-1 if state.current_player == 1 else 1)
            state.n += 1
        return v

    cdef int softmax_step(self, vector[float]* p):
        cdef float psum = 0
        for i in range(p.size()):
            psum += p[0][i]

        if psum > 0:
            for i in range(p.size()):
                p[0][i] /= psum
        else:
            for i in range(p.size()):
                p[0][i] = 1.0 / p.size()

        cdef float r = <float>rand() / RAND_MAX

        for i in range(p.size()):
            r -= p[0][i]
            if r <= 0:
                return i
        return p.size() - 1

    cdef int mcts_game_step(self, WattenEnv env, MCTSState* root, Model model, vector[float]* p, int steps=0):
        if steps == 0:
            steps = self.mcts_sims

        cdef int i, player
        for i in range(steps):
            self.mcts_sample(env, root, model, &player)

        cdef int child_n
        cdef float p_max = -2
        cdef int p_max_id = 0
        cdef float current_p
        p.clear()
        for i in range(root.childs.size()):
            p.push_back(self.calc_q(&root.childs[i], root.current_player, &child_n))

        #weights[]
        #for i in range(root.childs.size()):
        #    p[0][i] = (p_max - p[0][i] < 0.1)

        cdef vector[float] p_step
        for i in range(root.childs.size()):
            p_step.push_back(root.childs[i].p)

        return self.softmax_step(&p_step)


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

    cdef void mcts_game(self, WattenEnv env, Model model, Storage storage):
        cdef Observation obs
        cdef State game_state
        cdef Card* card
        cdef int last_player, i, j

        env.reset(&obs)
        if self.objective_opponent:
            env.current_player = rand() % 2

        cdef MCTSState root = self.create_root_state(env)
        cdef MCTSState tmp
        cdef vector[int] values
        cdef vector[float] p
        cdef int storage_index

        while not env.is_done():


            game_state = env.get_state()
            a = self.mcts_game_step(env, &root, model, &p)
            env.set_state(&game_state)

            j = 0
            for card in env.players[env.current_player].hand_cards:
                storage_index = storage.add_item()
                storage.data[storage_index].obs = obs

                storage.data[storage_index].output.v = 1 if env.current_player is 0 else -1
                storage.data[storage_index].weight_v = 1.0 / env.players[env.current_player].hand_cards.size()

                for i in range(32):
                    storage.data[storage_index].output.p[i] = (i == card.id)
                storage.data[storage_index].weight_p = (p[j] + 1) / 2

                values.push_back(storage_index)
                j += 1

            #print(np.array(storage.data[storage_index].obs.sets)[:,:,0])
            #print(storage.data[storage_index].output.p)

            last_player = env.current_player
            env.step(env.players[env.current_player].hand_cards[a].id, &obs)
            tmp = root.childs[a]
            root = tmp
            root.is_root = True

        for i in values:
            storage.data[i].output.v *= (1 if env.last_winner is 0 else -1)


    cpdef void mcts_generate(self, WattenEnv env, Model model, Storage storage):
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
        dot.write_svg('mcts-tree.svg')

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

    cpdef draw_game_tree(self, ModelRating rating, WattenEnv env, Model model, int game_index, int tree_depth, pre_actions):
        env.set_state(&rating.eval_games[game_index])
        env.current_player = 0

        for action in pre_actions:
            env.step(env.players[env.current_player].hand_cards[action].id)

        cdef MCTSState root = self.create_root_state(env)
        cdef vector[float] p

        a = self.mcts_game_step(env, &root, model, &p)
        self.draw_tree(&root, tree_depth)
        return p, a

    def end(self):
        pass