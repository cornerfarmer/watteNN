from libcpp.string cimport string
from libcpp cimport bool
from gym_watten.envs.watten_env cimport Observation, WattenEnv, Card
from src.MCTS cimport Storage
from src cimport ModelOutput
from src.Model cimport Model
from libc.stdlib cimport rand
import time
from src.TinyDnn cimport network, fc, sequential, construct_graph, input, shape3d, concat, layer, conv, connect, tensor_t, relu, sigmoid, tanh, mse, graph_visualizer, ofstream, vec_t
from libcpp.memory cimport make_shared, shared_ptr

cdef class TinyDnnModel(Model):
    def __init__(self, hidden_neurons=128):
        cdef int i

        self.input_layer.resize(2)
        self.input_layer[0].reset(new input(shape3d(1, 1, 192)))
        self.input_layer[1].reset(new input(shape3d(1, 1, 4)))

        cdef vector[shape3d] concat_shapes
        concat_shapes.push_back(shape3d(1, 1, 192))
        concat_shapes.push_back(shape3d(1, 1, 4))

        self.conv_layer.resize(2)
        self.conv_layer[0].reset(new conv(4, 8, 4, 8, 6, 192))
        self.conv_layer[1].reset(new concat(concat_shapes))
        #self.conv_layer[2].reset(new fc(196, 64))

       # connect(self.input_layer[0].get(), self.conv_layer[0].get(), 0, 0)
        connect(self.input_layer[0].get(), self.conv_layer[1].get(), 0, 0)
        connect(self.input_layer[1].get(), self.conv_layer[1].get(), 0, 1)
        #connect(self.conv_layer[1].get(), self.conv_layer[2].get(), 0, 0)

        self.policy_layer.resize(4)
        self.policy_layer[0].reset(new fc(196, hidden_neurons))
        self.policy_layer[1].reset(new relu())
        self.policy_layer[2].reset(new fc(hidden_neurons, 32))
        self.policy_layer[3].reset(new sigmoid())

        self.value_layer.resize(4)
        self.value_layer[0].reset(new fc(196, hidden_neurons))
        self.value_layer[1].reset(new relu())
        self.value_layer[2].reset(new fc(hidden_neurons, 1))
        self.value_layer[3].reset(new tanh())

        connect(self.conv_layer.back().get(), self.policy_layer[0].get(), 0, 0)
        for i in range(1, self.policy_layer.size()):
             connect(self.policy_layer[i - 1].get(), self.policy_layer[i].get(), 0, 0)

        connect(self.conv_layer.back().get(), self.value_layer[0].get(), 0, 0)
        for i in range(1, self.value_layer.size()):
             connect(self.value_layer[i - 1].get(), self.value_layer[i].get(), 0, 0)

        self.output_layer.push_back(shared_ptr[layer](self.policy_layer.back()))
        self.output_layer.push_back(shared_ptr[layer](self.value_layer.back()))

        construct_graph(self.model, self.input_layer, self.output_layer)

        self.model_input.resize(2)
        self.model_input[0].resize(4 * 8 * 6)
        self.model_input[1].resize(4)

        cdef ofstream ofs = ofstream("graph_net_example.txt")
        cdef graph_visualizer* viz = new graph_visualizer(self.model)
        viz.generate(ofs)
        self.timing = [0] * 3
        print(self.opt.alpha, self.opt.mu)

    cpdef void memorize_storage(self, Storage storage, bool clear_afterwards=True, int epochs=1, int number_of_samples=0):
        number_of_samples = max(number_of_samples, storage.number_of_samples)

        self.training_input.resize(storage.data.size() if number_of_samples is 0 else number_of_samples)
        self.training_output.resize(storage.data.size() if number_of_samples is 0 else number_of_samples)

        cdef int sample_index = 0
        for i in range(self.training_input.size()):
            self.training_input[i].resize(2)
            self.training_input[i][0].resize(4 * 8 * 6)
            self.training_input[i][1].resize(4)

            self.training_output[i].resize(2)
            self.training_output[i][0].resize(32)
            self.training_output[i][1].resize(1)

            if number_of_samples is 0:
                sample_index = i
            else:
                sample_index = rand() % storage.number_of_samples

            self._obs_to_tensor(&storage.data[sample_index].obs, &self.training_input[i])
            self._output_to_tensor(&storage.data[sample_index].output, &self.training_output[i])

        self.model.fit[mse](self.opt, self.training_input, self.training_output, 64, epochs)

        if clear_afterwards:
            storage.data.clear()

    cdef void _obs_to_tensor(self, Observation* obs, tensor_t* tensor):
        cdef int i, j, k
        for i in range(4):
            for j in range(8):
                for k in range(6):
                    tensor[0][0][k + i * 6 + j * 4 * 6] = obs.hand_cards[i][j][k]

        for i in range(4):
            tensor[0][1][i] = obs.tricks[i]

    cdef void _output_to_tensor(self, ModelOutput* output, tensor_t* tensor):
        cdef int i

        for i in range(32):
            tensor[0][0][i] = output.p[i]
        tensor[0][1][0] = output.v

    cdef void predict_single(self, Observation* obs, ModelOutput* output):
        cdef int i
        begin = time.time()
        self._obs_to_tensor(obs, &self.model_input)
        self.timing[0] += time.time() - begin

        begin = time.time()
        self.model_output = self.model.predict(self.model_input)
        self.timing[1] += time.time() - begin

        begin = time.time()
        for i in range(32):
            output.p[i] = self.model_output[0][i]
        output.v = self.model_output[1][0]
        self.timing[2] += time.time() - begin


    cdef void copy_weights_from(self, Model other_model):
        cdef int i,j
        cdef vector[vec_t*] own_weights
        cdef vector[vec_t*] other_weights
        for i in range(self.model.depth()):
            own_weights = self.model[i].weights()
            other_weights = (<TinyDnnModel>other_model).model[i].weights()

            for j in range(own_weights.size()):
                own_weights[j][0] = other_weights[j][0]