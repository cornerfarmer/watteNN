import unittest
from gym_watten.envs.watten_env cimport WattenEnv, Observation
from src.TinyDnnModel cimport TinyDnnModel, ModelOutput, Storage
from src cimport StorageItem
from libcpp.vector cimport vector
import time

class TinyDnnTest(unittest.TestCase):


    def test_memorize(self):
        cdef Storage storage = Storage()
        storage.data.push_back(StorageItem())

        cdef WattenEnv env = WattenEnv(True)
        env.seed(1850)
        env.reset(&storage.data.back().obs)

        cdef TinyDnnModel model = TinyDnnModel(env)

        for i in range(32):
            storage.data.back().output.p[i] = 0
        storage.data.back().output.p[4] = 1
        storage.data.back().output.v = -1

        model.memorize_storage(storage, clear_afterwards=False, epochs=1000)

        cdef ModelOutput prediction
        model.predict_single(&storage.data.back().obs, &prediction)
        print(prediction)
        for i in range(32):
            self.assertAlmostEqual(prediction.p[i], storage.data.back().output.p[i], 1, "Wrong probability memorized")
        self.assertAlmostEqual(prediction.v, storage.data.back().output.v, 1, "Wrong value memorized")

        cdef TinyDnnModel other_model = TinyDnnModel(env)
        other_model.copy_weights_from(model)

        cdef ModelOutput other_prediction
        other_model.predict_single(&storage.data.back().obs, &other_prediction)
        for i in range(32):
            self.assertEqual(other_prediction.p[i], prediction.p[i], "Probabilities not equal")
        self.assertEqual(other_prediction.v, prediction.v, "Value not equal")


    def test_predict_single(self):

        cdef WattenEnv env = WattenEnv()
        env.seed(1850)
        cdef Observation obs
        env.reset(&obs)
        cdef TinyDnnModel model = TinyDnnModel(env)
        cdef ModelOutput output

        for i in range(32):
            output.p[i] = 1
        output.v = 0

        begin = time.time()
        cdef ModelOutput prediction
        for i in range(1000):
            model.predict_single(&obs, &prediction)
        print(time.time() - begin)
        print(model.timing)

        for i in range(32):
            self.assertTrue(0.4 < prediction.p[i] < 0.6, "Prediction probability not in range")
        self.assertTrue(-0.2 < prediction.v < 0.2, "Prediction value not in range")