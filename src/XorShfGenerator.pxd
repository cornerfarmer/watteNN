


cdef class XorShfGenerator:
    cdef unsigned long x, y, z, seed

    cdef unsigned long min(self)
    cdef unsigned long max(self)
    cdef unsigned long rand(self)
    cpdef float randFloat(self)
    cdef void reset(self)