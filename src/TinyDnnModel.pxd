from libcpp.string cimport string
from libcpp.map cimport map
from libcpp.vector cimport vector
from libcpp cimport bool
from gym_watten.envs.watten_env cimport Observation, Card
from src.Storage cimport Storage
from src cimport ModelOutput, Experience
from src.Model cimport Model

from src.TinyDnn cimport network, graph, input, concat, layer_pointer, tensor_t, momentum

cdef class TinyDnnModel(Model):
    cdef network[graph] model
    cdef vector[layer_pointer] input_layer
    cdef vector[layer_pointer] conv_layer
    cdef vector[layer_pointer] output_layer
    cdef vector[layer_pointer] policy_layer
    cdef vector[layer_pointer] value_layer
    cdef tensor_t model_input
    cdef tensor_t model_output
    cdef vector[tensor_t] training_input
    cdef vector[tensor_t] training_output
    cdef momentum opt
    cdef object timing
    cdef int input_sets_size

    cpdef vector[float] memorize_storage(self, Storage storage, bool clear_afterwards=?, int epochs=?, int number_of_samples=?)
    cdef void _obs_to_tensor(self, Observation* obs, tensor_t* tensor)
    cdef void _output_to_tensor(self, ModelOutput* output, tensor_t* tensor)
    cdef void predict_single(self, Observation* obs, ModelOutput* output)
    cdef int valid_step(self, float* values, vector[Card*]* hand_cards)
    cdef int argmax(self, vector[float]* values)
    cpdef void copy_weights_from(self, Model other_model)
    cdef void print_weights(self)