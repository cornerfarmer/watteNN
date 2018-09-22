from gym_watten.envs.watten_env cimport State, Observation
from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.string cimport string

cdef struct ModelOutput:
    float p[32]
    float v
    float scale

cdef struct Experience:
    ModelOutput output
    int n

cdef cppclass MCTSState:
    vector[MCTSState*] childs
    float n
    float w
    float v
    float p
    State env_state
    float end_v
    int current_player
    bool is_root
    float scale
    MCTSState* parent
    bool is_leaf

cdef struct StorageItem:
    Observation obs
    ModelOutput output
    float weight
    bool value_net
    string key
    float equalizer_weight
