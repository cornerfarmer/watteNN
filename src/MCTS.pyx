from gym_watten.envs.watten_env cimport State, WattenEnv, Card, Observation
from libcpp cimport bool
from libcpp.vector cimport vector
from libc.math cimport sqrt, exp
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
import time as pytime
from libcpp.string cimport string

cdef extern from "<string>" namespace "std":
    string to_string(int val)

cdef class PredictionQueue:
    def __cinit__(self):
        pass

    cdef enqueue(self, Observation* full_obs, Observation* obs, int worker_id):
        self.obs_queue.push_back(obs[0])
        self.full_obs_queue.push_back(full_obs[0])
        self.outputs.push_back(ModelOutput())
        self.worker_ids.push_back(worker_id)

    cdef do_prediction(self, WattenEnv env, Model model, object worker):
        model.predict(&self.full_obs_queue, &self.obs_queue, &self.outputs)

        cdef int worker_id
        cdef int i = 0
        cdef MCTSWorker current_worker
        for worker_id in self.worker_ids:
            current_worker = <MCTSWorker>(worker[worker_id])
            current_worker.handle_prediction(env, &self.outputs[i])
            i += 1

        self.obs_queue.clear()
        self.worker_ids.clear()
        self.full_obs_queue.clear()
        self.outputs.clear()

cdef class MCTSWorker:

    def __cinit__(self, worker_id, rng, mcts_sims=20, exploration=0.1, only_one_step=False, step_exploration=0.1):
        self.worker_id = worker_id
        self.mcts_sims = mcts_sims
        self.exploration = exploration
        self.current_step = 0
        self.prediction_state = NULL
        self.only_one_step = only_one_step
        self.value_storage = Storage()
        self.step_exploration = step_exploration
        self.exploration_mode.resize(2)
        self.rng = rng

    def __dealloc__(self):
        self._clear_nodes()

    cdef void _clear_nodes(self):
        for i in range(self.nodes.size()):
            del self.nodes[i]
        self.nodes.clear()

    cdef void reset(self, env):
        self._clear_nodes()
        self.nodes.push_back(new MCTSState())
        self.create_root_state(env, self.nodes[0])
        self.root = self.nodes[0]
        self.current_step = 0
        self.prediction_state = NULL
        self.finished = False
        self.value_storage.clear()
        self.exploration_mode[0] = False
        self.exploration_mode[1] = False

    cdef void add_state(self, MCTSState* parent, float p, WattenEnv env, int end_v=0):
        self.nodes.push_back(new MCTSState())
        self.nodes.back().n = 0
        self.nodes.back().w = 0
        self.nodes.back().v = 0
        self.nodes.back().p = p
        self.nodes.back().env_state = env.get_state()
        self.nodes.back().current_player = env.current_player
        self.nodes.back().end_v = end_v
        self.nodes.back().is_root = False
        self.nodes.back().parent = parent
        self.nodes.back().is_leaf = True
        self.nodes.back().fully_explored = False
        parent.childs.push_back(self.nodes.back())

    cdef bool is_state_leaf_node(self, MCTSState* state):
        return state.childs.size() == 0

    cdef void revert_leaf_state(self, MCTSState* state):

        cdef float p = 1
        cdef float v = state.v
        while True:
            state.w -= v * p
            state.n -= p
            p *= state.p

            if state.is_root:
                break
            else:
                state = <MCTSState*>state.parent


    cdef void handle_leaf_state(self, MCTSState* state, float v):
        cdef float p = 1
        if state.n == 0:
            if not state.is_root and state.parent.is_leaf:
                self.revert_leaf_state(state.parent)
                state.parent.is_leaf = False

            while True:
                state.w += v * p
                state.n += p
                p *= state.p

                if state.is_root:
                    break
                else:
                    state = <MCTSState*>state.parent

    cdef void handle_prediction(self, WattenEnv env, ModelOutput* prediction):
        env.set_state(&self.prediction_state.env_state)

        cdef vector[Card*] hand_cards = env.players[env.current_player].hand_cards
        current_player = env.current_player
        for card in hand_cards:
            env.step(card.id)
            self.add_state(self.prediction_state, prediction.p[card.id], env, 0 if not env.is_done() else (1 if env.last_winner == 0 else -1))
            env.set_state(&self.prediction_state.env_state)

        v = prediction.v * (-1 if self.prediction_state.current_player == 1 else 1)
        self.prediction_state.v = v
        self.handle_leaf_state(self.prediction_state, v)

    cdef bool mcts_sample(self, WattenEnv env, MCTSState* state, PredictionQueue queue):
        cdef int current_player, i, a, childs_to_explore
        cdef float left_p
        cdef vector[float] p
        cdef bool finished
        cdef MCTSState* max_child

        if self.is_state_leaf_node(state):
            if state.end_v != 0:
                self.handle_leaf_state(state, state.end_v)
                finished = True
                state.fully_explored = True
            else:
                env.set_state(&state.env_state)

                env.regenerate_obs(&self._obs)
                env.regenerate_full_obs(&self._full_obs)
                queue.enqueue(&self._full_obs, &self._obs, self.worker_id)

                self.prediction_state = state
                finished = False
        else:
            childs_to_explore = 0
            left_p = 0
            for i in range(state.childs.size()):
                if state.childs[i].fully_explored:
                    p.push_back(0)
                    left_p += state.childs[i].p
                else:
                    p.push_back(state.childs[i].p)
                    childs_to_explore += 1

            if childs_to_explore > 0 and left_p > 0:
                left_p /= childs_to_explore
                for i in range(state.childs.size()):
                    if not state.childs[i].fully_explored:
                        p[i] += left_p


            if state.is_root and self.rng.randFloat() < self.exploration:
                a = self.rng.rand() % p.size()
            else:
                a = self.softmax_step(&p)

            max_child = state.childs[a]

            finished = self.mcts_sample(env, max_child, queue)

            if finished:

                state.fully_explored = True
                for i in range(state.childs.size()):
                    if not state.childs[i].fully_explored:
                        state.fully_explored = False
                        break


        return finished

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

        cdef float r = self.rng.randFloat()

        for i in range(p.size()):
            r -= p[0][i]
            if r <= 0:
                return i
        return p.size() - 1

    cdef bool mcts_game_step(self, WattenEnv env, PredictionQueue queue, vector[float]* p, float* v, int* action, bool* exploration_mode_activated):

        cdef int i
        cdef bool finished
    #if not self.exploration_mode or self.root.current_player == self.exploration_player:
        for i in range(self.current_step, self.mcts_sims):
            finished = self.mcts_sample(env, self.root, queue)
            self.current_step += 1
            if not finished:
                return False
            elif self.root.fully_explored:
                break
        self.current_step = 0

        p.clear()
        for i in range(self.root.childs.size()):
            p.push_back((self.root.childs[i].w / self.root.childs[i].n * (-1 if self.root.current_player == 1 else 1)) if self.root.childs[i].n > 0 else -1)

        v[0] = 0
        for i in range(self.root.childs.size()):
            if self.root.childs[i].end_v != 0:
                v[0] += self.root.childs[i].p * self.root.childs[i].end_v
            else:
                v[0] += self.root.childs[i].p * self.root.childs[i].v
        v[0] *= (-1 if self.root.current_player == 1 else 1)
        v[0] = self.root.w / self.root.n * (-1 if self.root.current_player == 1 else 1)

        cdef vector[float] p_step
        cdef int number_of_zero_p = 0
        for i in range(self.root.childs.size()):
            p_step.push_back(self.root.childs[i].p)
            if p_step.back() == 0:
                number_of_zero_p += 1

        cdef int r
        if number_of_zero_p > 0 and self.rng.randFloat() < self.step_exploration:
            r = self.rng.rand() % number_of_zero_p

            for i in range(p_step.size()):
                if p_step[i] == 0:
                    if r == 0:
                        action[0] = i
                        break
                    else:
                        r -= 1

            self.exploration_mode[self.root.current_player] = True
            self.exploration_player = self.root.current_player
            exploration_mode_activated[0] = True
        else:
            action[0] = self.softmax_step(&p_step)
            exploration_mode_activated[0] = False

        return True


    cdef void create_root_state(self, WattenEnv env, MCTSState* state):
        state.n = 0
        state.w = 0
        state.v = 0
        state.p = 1
        state.env_state = env.get_state()
        state.current_player = env.current_player
        state.end_v = 0
        state.is_root = True
        state.is_leaf = True
        state.fully_explored = False

    cdef bool mcts_game(self, WattenEnv env, PredictionQueue queue, Storage storage):
        cdef Observation obs, full_obs
        cdef Card* card
        cdef int i, j

        cdef MCTSState tmp

        cdef vector[float] p
        cdef int storage_index, a
        cdef float v
        cdef env_is_done = False
        cdef string key
        cdef bool exploration_mode_activated
        cdef bool exploration_mode_active

        while not env_is_done:
            finished = self.mcts_game_step(env, queue, &p, &v, &a, &exploration_mode_activated)
            if not finished:
                return False
            env.set_state(&self.root.env_state)

            env.regenerate_obs(&obs)
            env.regenerate_full_obs(&full_obs)

            storage_index = self.value_storage.add_item()
            self.value_storage.data[storage_index].obs = full_obs
            self.value_storage.data[storage_index].output.v = v
            self.value_storage.data[storage_index].value_net = True

            exploration_mode_active = self.exploration_mode[0] or self.exploration_mode[1]

            if not exploration_mode_active or not self.exploration_mode[1 - env.current_player]:
                j = 0
                for card in env.players[env.current_player].hand_cards:
                    if p[j] > -1:
                        storage_index = storage.add_item()
                        storage.data[storage_index].obs = obs
                        for i in range(32):
                            storage.data[storage_index].output.p[i] = (i == card.id)
                        storage.data[storage_index].weight = (p[j] + 1) / 2

                        storage.data[storage_index].value_net = False
                        #if obs.sets[0][4][0] == 1 and obs.sets[1][4][0] == 1 and obs.sets[1][7][0] == 1 and obs.sets[1][5][1] == 1:
                        #if obs.sets[0][5][0] == 1 and obs.sets[1][5][0] == 1 and obs.sets[1][6][2] == 1 and obs.sets[0][4][3] == 1:
                        #if obs.sets[0][4][0] == 1 and obs.sets[1][4][0] == 1 and obs.sets[0][7][2] == 1 and obs.sets[1][5][3] == 1:
                        #    print(storage.data[storage_index].weight, storage.data[storage_index].output.p, p)
                    j += 1

            env.step(env.players[env.current_player].hand_cards[a].id, &obs)
            self.root = self.root.childs[a]
            self.root.is_root = True

            env_is_done =  env.is_done()

            if self.only_one_step:
                break

        #for i in range(self.value_storage.number_of_samples):
        #    self.value_storage.data[i].output.v *= (1 if env.last_winner is 0 else -1)

        storage.copy_from(self.value_storage)

        self.finished = True
        return True

cdef class MCTS:

    def __cinit__(self, rng, episodes=75, mcts_sims=20, exploration=0.1, only_one_step=False, step_exploration=0.1):
        self.episodes = episodes
        self.worker = []
        self.rng = rng
        for i in range(episodes):
            self.worker.append(MCTSWorker(i, self.rng, mcts_sims, exploration, only_one_step, step_exploration))

    cpdef void mcts_generate(self, WattenEnv env, Model model, Storage storage, ModelRating rating, bool reset_env=True):
        cdef PredictionQueue queue = PredictionQueue()
        cdef int i
        cdef MCTSWorker current_worker
        for i in range(len(self.worker)):
            current_worker = <MCTSWorker>(self.worker[i])
            if reset_env:
                env.reset()
                #env.set_state(&rating.eval_games[230])
            current_worker.reset(env)

        cdef bool finished = False
        while not finished:
            finished = True
            for i in range(len(self.worker)):
                current_worker = <MCTSWorker>(self.worker[i])
                if not current_worker.finished:
                    current_worker.mcts_game(env, queue, storage)
                    if not current_worker.finished:
                        finished = False

            if not finished:
                queue.do_prediction(env, model, self.worker)


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
        text += "W: " + str(root.w) + '\n'
        text += "P: " + str(root.p) + '\n'
        text += "V: " + str(root.v)

        node = pydot.Node(str(id), label=text)
        dot.add_node(node)
        id += 1

        if tree_depth > 1:
            i = 0
            for i in range(root.childs.size()):
                if len(tree_path) == 0 or (type(tree_path[0]) == list and i in tree_path[0] or type(tree_path[0]) == int and tree_path[0] == i):
                    child_node, id = self.create_nodes(root.childs[i], dot, tree_depth - 1, tree_path if len(tree_path) == 0 else tree_path[1:], id)
                    dot.add_edge(pydot.Edge(node.get_name(), child_node.get_name()))
                i+=1

        return node, id

    cpdef draw_game_tree(self, ModelRating rating, WattenEnv env, Model model, Storage storage, int game_index, int tree_depth, pre_actions):
        env.set_state(&rating.eval_games[game_index])
        env.current_player = 0

        for action in pre_actions:
            env.step(env.players[env.current_player].hand_cards[action].id)

        cdef Observation obs
        env.regenerate_obs(&obs)
        print(obs)
        start = pytime.time()
        self.mcts_generate(env, model, storage, rating, False)
        print(pytime.time() - start)

        self.draw_tree((<MCTSWorker>(self.worker[0])).nodes[0], tree_depth)

    cpdef create_storage(self, ModelRating rating, WattenEnv env, Model model, Storage storage, int game_index):
        env.set_state(&rating.eval_games[game_index])
        env.current_player = 0

        start = pytime.time()
        self.mcts_game(env, model, storage, False)
        end = pytime.time()
        print(end - start)
        print(self.timing)

        #for i in range(storage.data.size()):
        #    print(storage.data[i])

    def end(self):
        pass