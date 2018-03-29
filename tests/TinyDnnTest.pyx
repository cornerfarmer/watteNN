import unittest
from gym_watten.envs.watten_env cimport WattenEnv, Observation
from src.TinyDnnModel cimport TinyDnnModel, ModelOutput
from src cimport StorageItem
from libcpp.vector cimport vector

class TinyDnnTest(unittest.TestCase):

    def test_init(self):

        cdef WattenEnv env = WattenEnv()
        env.seed(1850)
        cdef Observation obs
        env.reset(&obs)
        cdef TinyDnnModel model = TinyDnnModel()
        cdef ModelOutput output

        for i in range(32):
            output.p[i] = 1
        output.v = 0

        cdef ModelOutput prediction
        model.predict_single(&obs, &prediction)

        #for i in range(32):
        #    self.assertTrue(0.4 < prediction.p[i] < 0.6, "Prediction probability not in range")
        #self.assertTrue(-0.2 < prediction.v < 0.2, "Prediction value not in range")