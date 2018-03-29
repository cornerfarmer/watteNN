from libcpp.string cimport string
from libcpp.map cimport map
from libcpp.vector cimport vector
from libcpp cimport bool
from gym_watten.envs.watten_env cimport Observation, Card
from src.Storage cimport Storage
from src cimport ModelOutput, Experience
from src.Model cimport Model

from src.TinyDnn cimport network, graph, input, concat, layer_pointer

cdef class TinyDnnModel(Model):
    cdef network[graph] model
    cdef vector[layer_pointer] input_layer
    cdef vector[layer_pointer] hidden_layer
    cdef vector[layer_pointer] output_layer

    cpdef void memorize_storage(self, Storage storage, bool clear_afterwards=?, int epochs=?)
    cdef void predict_single(self, Observation* obs, ModelOutput* output)
    cdef int valid_step(self, float* values, vector[Card*]* hand_cards)
    cdef int argmax(self, vector[float]* values)
    cdef void copy_weights_from(self, Model other_model)