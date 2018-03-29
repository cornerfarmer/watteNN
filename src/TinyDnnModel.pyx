from libcpp.string cimport string
from libcpp cimport bool
from gym_watten.envs.watten_env cimport Observation, WattenEnv, Card
from src.MCTS cimport Storage
from src cimport ModelOutput
from src.Model cimport Model

from src.TinyDnn cimport network, fc, sequential, construct_graph, input, shape3d, concat, layer, conv, connect, tensor_t
from libcpp.memory cimport make_shared

cdef class TinyDnnModel(Model):
    def __init__(self):

        print("1")
        self.input_layer.resize(2)
        self.input_layer[0].reset(new input(shape3d(4, 8, 6)))
        self.input_layer[1].reset(new input(shape3d(1, 1, 4)))
        print("2")

        cdef vector[shape3d] concat_shapes
        concat_shapes.push_back(shape3d(1, 1, 192))
        concat_shapes.push_back(shape3d(1, 1, 4))
        self.hidden_layer.resize(2)
        self.hidden_layer[0].reset(new conv(4, 8, 4, 8, 6, 192))
        self.hidden_layer[1].reset(new concat(concat_shapes))

        self.output_layer.resize(2)
        self.output_layer[0].reset(new fc(196, 32))
        self.output_layer[1].reset(new fc(196, 1))
        print("3")
        self.input_layer[0].get()[0] << self.hidden_layer[0].get()[0] << self.hidden_layer[1].get()[0]
        print("4")
        connect(self.input_layer[1].get(), self.hidden_layer[1].get(), 0, 1)

        self.hidden_layer[1].get()[0] << self.output_layer[0].get()[0]
        self.hidden_layer[1].get()[0] << self.output_layer[1].get()[0]

        print("5")
        construct_graph(self.model, self.input_layer, self.output_layer)

        print(self.model.in_data_size(), self.model.out_data_size())

    cpdef void memorize_storage(self, Storage storage, bool clear_afterwards=True, int epochs=1):

        if clear_afterwards:
            storage.data.clear()

    cdef void predict_single(self, Observation* obs, ModelOutput* output):
        cdef int i
        cdef tensor_t model_input
        cdef tensor_t model_output

        model_input.resize(2)
        model_input[0].resize(4 * 8 * 6)
        model_input[1].resize(4)

        model_output = self.model.predict(model_input)

        print(model_output[0].size(), model_output[1].size())
        for i in range(32):
            print(<float>model_output[0][i])
        print(<float>model_output[1][0])



    cdef void copy_weights_from(self, Model other_model):
        pass