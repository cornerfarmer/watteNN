from gym_watten.envs.watten_env cimport State, Observation
from libcpp cimport bool
from libcpp.vector cimport vector

cdef struct ModelOutput:
    float p[32]
    float v
    float scale

cdef struct Experience:
    ModelOutput output
    int n

cdef cppclass MCTSState:
    vector[MCTSState*] childs
    int n
    float w
    float v
    float p
    State env_state
    float end_v
    int current_player
    bool is_root
    float scale
    MCTSState* parent

cdef struct StorageItem:
    Observation obs
    ModelOutput output
    float weight
    bool value_net
