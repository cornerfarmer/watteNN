from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr
cdef extern from "tiny_dnn/tiny_dnn.h" namespace "tiny_dnn" nogil:
    cdef cppclass shape3d:
        shape3d(size_t width, size_t height, size_t depth)

    cdef cppclass layer:
        layer & operator<<(layer &rhs)

    void connect(layer *head, layer *tail, size_t head_index, size_t tail_index)

    ctypedef vector[float] vec_t;
    ctypedef vector[vec_t] tensor_t;

cdef extern from "tiny_dnn/tiny_dnn.h" namespace "tiny_dnn::layers" nogil:
    cdef cppclass fc(layer):
        fc(size_t in_dim, size_t out_dim)

    cdef cppclass input(layer):
        input(const shape3d &shape)

    cdef cppclass concat(layer):
        concat(const vector[shape3d] &in_shapes)

    cdef cppclass conv(layer):
        conv(size_t in_width, size_t in_height, size_t window_width, size_t window_height, size_t in_channels, size_t out_channels)

ctypedef shared_ptr[layer] layer_pointer
cdef extern from "tiny_dnn/tiny_dnn.h" namespace "tiny_dnn" nogil:
    cdef cppclass sequential:
        pass

    cdef cppclass network[T]:
        network()
        size_t out_data_size() const
        size_t in_data_size() const

        network[T]& operator<<[L](L &&)

        tensor_t predict(const tensor_t &inp)
        vector[tensor_t] predict(const vector[tensor_t] &inp)

    cdef cppclass graph:
        pass

    void construct_graph(network[graph] &graph, const vector[layer_pointer] &inputs, const vector[layer_pointer] &outputs)

