from gym_watten.envs.watten_env cimport State, Observation
from libcpp cimport bool
from libcpp.vector cimport vector

cdef struct ModelOutput:
    float p[32]
    float v

cdef struct Experience:
    ModelOutput output
    int n

cdef struct MCTSState:
    vector[MCTSState] childs
    int n
    float w
    float v
    float p
    State env_state
    float end_v
    int current_player
    bool is_root

cdef struct StorageItem:
    Observation obs
    ModelOutput output
    float weight
    bool value_net
