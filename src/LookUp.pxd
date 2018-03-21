from libcpp.string cimport string
from libcpp.map cimport map
from gym_watten.envs.watten_env cimport Observation

cdef struct ModelOutput:
    float p[32]
    float v

cdef struct Experience:
    ModelOutput output
    int n

cdef class LookUp:
    cdef map[string, Experience] table

    cdef string generate_key(self, Observation* obs)
    cdef void memorize(self, Observation* obs, ModelOutput* value)
    cdef ModelOutput predict_single(self, Observation* obs)
