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
from libc.stdlib cimport srand
from libc.time cimport time
import time as pytime
from cpython.mem cimport PyMem_Malloc, PyMem_Free

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

    def __cinit__(self, worker_id, mcts_sims=20, exploration=0.1, only_one_step=False):
        self.worker_id = worker_id
        self.mcts_sims = mcts_sims
        self.exploration = exploration
        self.current_step = 0
        self.prediction_state = NULL
        self.only_one_step = only_one_step

    def __dealloc__(self):
        self._clear_nodes()

    cdef void _clear_nodes(self):
        for i in range(self.nodes.size()):
            PyMem_Free(self.nodes[i])
        self.nodes.clear()

    cdef void reset(self, env):
        self._clear_nodes()
        self.nodes.push_back(new MCTSState())
        self.nodes[0][0] = self.create_root_state(env)
        self.root = self.nodes[0]
        self.current_step = 0
        self.prediction_state = NULL
        self.finished = False
        self.values.clear()

    cdef void add_state(self, MCTSState* parent, float p, WattenEnv env, int end_v=0, float scale=1):
        self.nodes.push_back(new MCTSState())
        self.nodes.back().n = 0
        self.nodes.back().w = 0
        self.nodes.back().v = 0
        self.nodes.back().p = p
        self.nodes.back().env_state = env.get_state()
        self.nodes.back().current_player = env.current_player
        self.nodes.back().end_v = end_v
        self.nodes.back().is_root = False
        self.nodes.back().scale = scale
        self.nodes.back().parent = parent
        parent.childs.push_back(self.nodes.back())

    cdef bool is_state_leaf_node(self, MCTSState* state):
        return state.childs.size() == 0

    cdef void handle_leaf_state(self, MCTSState* state, float v):
        while True:
            state.w += v
            state.n += 1

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
            self.add_state(self.prediction_state, prediction.p[card.id], env, 0 if not env.is_done() else (1 if env.last_winner == 0 else -1), prediction.scale)
            env.set_state(&self.prediction_state.env_state)

        v = prediction.v * (-1 if self.prediction_state.current_player == 1 else 1)
        self.prediction_state.v = v
        self.handle_leaf_state(self.prediction_state, v)

    cdef bool mcts_sample(self, WattenEnv env, MCTSState* state, PredictionQueue queue):
        cdef int current_player, i
        cdef vector[float] p
        cdef bool finished
        cdef MCTSState* max_child
        if self.is_state_leaf_node(state):
            if state.end_v != 0:
                self.handle_leaf_state(state, state.end_v)
                finished = True
            else:
                env.set_state(&state.env_state)

                env.regenerate_obs(&self._obs)
                env.regenerate_full_obs(&self._full_obs)
                queue.enqueue(&self._full_obs, &self._obs, self.worker_id)

                self.prediction_state = state
                finished = False
        else:
            for i in range(state.childs.size()):
                p.push_back(state.childs[i].p)
            max_child = state.childs[self.softmax_step(&p, state.is_root)]

            finished = self.mcts_sample(env, max_child, queue)
        return finished

    cdef int softmax_step(self, vector[float]* p, bool do_exploration=False):

        if do_exploration and <float>rand() / RAND_MAX < self.exploration:
            return rand() % p.size()

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

    cdef bool mcts_game_step(self, WattenEnv env, PredictionQueue queue, vector[float]* p, float* scale, int* action):

        cdef int i
        cdef bool finished
        for i in range(self.current_step, self.mcts_sims):
            finished = self.mcts_sample(env, self.root, queue)
            self.current_step += 1
            if not finished:
                return False
        self.current_step = 0

        p.clear()
        for i in range(self.root.childs.size()):
            p.push_back((self.root.childs[i].w / self.root.childs[i].n * (-1 if self.root.current_player == 1 else 1)) if self.root.childs[i].n > 0 else -1)
        scale[0] = self.root.childs[0].scale

        cdef vector[float] p_step
        for i in range(self.root.childs.size()):
            p_step.push_back(self.root.childs[i].p)

        action[0] = self.softmax_step(&p_step, False)
        return True


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

    cdef bool mcts_game(self, WattenEnv env, PredictionQueue queue, Storage storage):
        cdef Observation obs, full_obs
        cdef Card* card
        cdef int i, j

        cdef MCTSState tmp

        cdef vector[float] p
        cdef int storage_index, a
        cdef float scale
        cdef env_is_done = False

        while not env_is_done:
            finished = self.mcts_game_step(env, queue, &p, &scale, &a)
            if not finished:
                return False
            env.set_state(&self.root.env_state)

            env.regenerate_obs(&obs)
            env.regenerate_full_obs(&full_obs)
            storage_index = storage.add_item()
            storage.data[storage_index].obs = full_obs
            storage.data[storage_index].output.v = 1 if env.current_player is 0 else -1
            storage.data[storage_index].value_net = True
            self.values.push_back(storage_index)

            j = 0
            for card in env.players[env.current_player].hand_cards:
                if p[j] > -1:
                    storage_index = storage.add_item()
                    storage.data[storage_index].obs = obs
                    for i in range(32):
                        storage.data[storage_index].output.p[i] = (i == card.id)
                    storage.data[storage_index].weight = (p[j] + 1) / 2 * 1 / scale
                    storage.data[storage_index].value_net = False
                    storage.data[storage_index].output.scale = (p[j] + 1) / 2
                    if obs.sets[0][4][0] == 1 and obs.sets[1][4][0] == 1 and obs.sets[1][7][0] == 1 and obs.sets[1][5][1] == 1:
                        print(storage.data[storage_index].weight, storage.data[storage_index].output.p)
                j += 1

            env.step(env.players[env.current_player].hand_cards[a].id, &obs)
            self.root = self.root.childs[a]
            self.root.is_root = True

            env_is_done =  env.is_done()

            if self.only_one_step:
                break

        for i in self.values:
            storage.data[i].output.v *= (1 if env.last_winner is 0 else -1)

        self.finished = True
        return True

cdef class MCTS:

    def __cinit__(self, episodes=75, mcts_sims=20, exploration=0.1, only_one_step=False):
        self.episodes = episodes
        self.worker = []
        for i in range(episodes):
            self.worker.append(MCTSWorker(i, mcts_sims, exploration, only_one_step))

    cpdef void mcts_generate(self, WattenEnv env, Model model, Storage storage, bool reset_env=True):
        cdef PredictionQueue queue = PredictionQueue()
        cdef int i
        cdef MCTSWorker current_worker
        for i in range(len(self.worker)):
            current_worker = <MCTSWorker>(self.worker[i])
            if reset_env:
                env.reset()
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

        srand(time(NULL))

        for action in pre_actions:
            env.step(env.players[env.current_player].hand_cards[action].id)

        start = pytime.time()
        self.mcts_generate(env, model, storage, False)
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