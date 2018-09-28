from libc.time cimport time
from libc.stdlib cimport srand
from libc.stdlib cimport rand, RAND_MAX

cdef class XorShfGenerator:
    def __cinit__(self, seed=-1):
        if seed == -1:
            srand(time(NULL))
            self.seed = <unsigned long>(rand() / <float>RAND_MAX * <unsigned long>(-1))
        else:
            self.seed = seed
        self.reset()

    cdef unsigned long min(self):
        return 0

    cdef unsigned long max(self):
        return -1

    cdef unsigned long rand(self):
        cdef unsigned long t

        self.x ^= self.x << 16
        self.x ^= self.x >> 5
        self.x ^= self.x << 1

        t = self.x
        self.x = self.y
        self.y = self.z
        self.z = t ^ self.x ^ self.y

        return self.z

    cpdef float randFloat(self):
        return (float)(self.rand() - self.min()) / self.max()

    cdef void reset(self):
        self.x = self.seed
        self.y = 15204331779905043239
        self.z = 2290907190372362775
