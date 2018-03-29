from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.memory cimport shared_ptr
from libcpp cimport bool

cdef extern from "<fstream>" namespace "std" nogil:
    cdef cppclass ofstream:
        ofstream()
        ofstream(string)

cdef extern from "tiny_dnn/tiny_dnn.h" namespace "tiny_dnn" nogil:
    ctypedef vector[float] vec_t;
    ctypedef vector[vec_t] tensor_t;

    cdef cppclass shape3d:
        shape3d(size_t width, size_t height, size_t depth)

    cdef cppclass layer:
        layer & operator<<(layer &rhs)
        vector[vec_t*] weights()

    void connect(layer *head, layer *tail, size_t head_index, size_t tail_index)


cdef extern from "tiny_dnn/tiny_dnn.h" namespace "tiny_dnn::layers" nogil:
    cdef cppclass fc(layer):
        fc(size_t in_dim, size_t out_dim)

    cdef cppclass input(layer):
        input(const shape3d &shape)

    cdef cppclass concat(layer):
        concat(const vector[shape3d] &in_shapes)

    cdef cppclass conv(layer):
        conv(size_t in_width, size_t in_height, size_t window_width, size_t window_height, size_t in_channels, size_t out_channels)

cdef extern from "tiny_dnn/tiny_dnn.h" namespace "tiny_dnn::activation" nogil:
    cdef cppclass relu(layer):
        relu()

    cdef cppclass tanh(layer):
        tanh()

    cdef cppclass sigmoid(layer):
        sigmoid()

ctypedef shared_ptr[layer] layer_pointer
cdef extern from "tiny_dnn/tiny_dnn.h" namespace "tiny_dnn" nogil:
    cdef cppclass sequential:
        pass

    cdef cppclass optimizer:
        pass

    cdef cppclass network[T]:
        network()
        size_t out_data_size() const
        size_t in_data_size() const

        network[T]& operator<<[L](L &&)

        tensor_t predict(const tensor_t &inp)
        vector[tensor_t] predict(const vector[tensor_t] &inp)
        bool fit[Error](optimizer& opt, const vector[tensor_t]& inputs, const vector[tensor_t]& desired_outputs, size_t batch_size, int epoch)
        void save(string f)
        void load(string f)
        size_t depth()
        layer* operator[](size_t index)
        bool has_same_weights(const network[T] &rhs, float eps)

    cdef cppclass graph:
        pass

    void construct_graph(network[graph] &graph, const vector[layer_pointer] &inputs, const vector[layer_pointer] &outputs)

    cdef cppclass adam(optimizer):
        pass

    cdef cppclass momentum(optimizer):
        float alpha
        float mu

    cdef cppclass mse:
        pass

    cdef cppclass cross_entropy_multiclass:
        pass


    cdef cppclass graph_visualizer:
        graph_visualizer()
        graph_visualizer(network[graph])

        void generate(ofstream& stream)