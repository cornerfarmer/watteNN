import unittest
from gym_watten.envs.watten_env cimport WattenEnv, Observation
from src.TinyDnnModel cimport TinyDnnModel, ModelOutput, Storage
from src.KerasModel cimport KerasModel
from src.ModelRating cimport ModelRating
from src cimport StorageItem
from libcpp.vector cimport vector
from src.MCTS cimport MCTS
from libc.stdlib cimport srand
import time

class TinyDnnTest(unittest.TestCase):


    def test_memorize(self):
        cdef Storage storage = Storage()
        storage.data.push_back(StorageItem())

        cdef WattenEnv env = WattenEnv()
        env.seed(1850)
        env.reset(&storage.data.back().obs)

        cdef TinyDnnModel tinyModel = TinyDnnModel()
        cdef KerasModel kerasModel = KerasModel()

        cdef ModelRating rating = ModelRating(env)

        cdef MCTS mcts = MCTS(episodes=1)
        cdef Storage tinyStorage = Storage()
        srand(42)
        mcts.mcts_generate(env, tinyModel, tinyStorage)
        cdef Storage kerasStorage = Storage()
        srand(42)
        mcts.mcts_generate(env, kerasModel, kerasStorage)



        cdef int train_sample_size = 0
        srand(43)
        mcts.mcts_generate(env, tinyModel, tinyStorage)
        srand(43)
        mcts.mcts_generate(env, kerasModel, kerasStorage)

       # kerasStorage.data.resize(2)
        #kerasStorage.number_of_samples = 2
        #tinyStorage.data.resize(2)
        #tinyStorage.number_of_samples = 2

        tinyModel.memorize_storage(tinyStorage, train_sample_size == 0, 1, train_sample_size)
        kerasModel.memorize_storage(kerasStorage, train_sample_size == 0, 1, train_sample_size)

        srand(43)
        mcts.mcts_generate(env, tinyModel, tinyStorage)
        srand(43)
        mcts.mcts_generate(env, kerasModel, kerasStorage)

        print(tinyStorage.data[0], tinyStorage.data.size())
        print(kerasStorage.data[0], kerasStorage.data.size())
        """

        for i in range(32):
            storage.data.back().output.p[i] = 0
        storage.data.back().output.p[4] = 1
        storage.data.back().output.v = -1

        cdef ModelOutput kerasPrediction, tinyPrediction
        kerasModel.predict_single(&storage.data.back().obs, &kerasPrediction)
        tinyModel.predict_single(&storage.data.back().obs, &tinyPrediction)

        kerasModel.memorize_storage(storage, clear_afterwards=False, epochs=2)
        tinyModel.memorize_storage(storage, clear_afterwards=False, epochs=2)

        kerasModel.memorize_storage(storage, clear_afterwards=False, epochs=2)
        tinyModel.memorize_storage(storage, clear_afterwards=False, epochs=2)

        kerasModel.memorize_storage(storage, clear_afterwards=False, epochs=2)
        tinyModel.memorize_storage(storage, clear_afterwards=False, epochs=2)

        kerasModel.memorize_storage(storage, clear_afterwards=False, epochs=2)
        tinyModel.memorize_storage(storage, clear_afterwards=False, epochs=2)

        kerasModel.predict_single(&storage.data.back().obs, &kerasPrediction)
        tinyModel.predict_single(&storage.data.back().obs, &tinyPrediction)
        print(kerasPrediction)
        print(tinyPrediction)

        cdef KerasModel kerasModel2 = KerasModel()
        kerasModel2.copy_weights_from(kerasModel)
        kerasModel.predict_single(&storage.data.back().obs, &kerasPrediction)
        print(kerasPrediction)



        cdef ModelOutput prediction
        model.predict_single(&storage.data.back().obs, &prediction)
        print(prediction)
        for i in range(32):
            self.assertAlmostEqual(prediction.p[i], storage.data.back().output.p[i], 1, "Wrong probability memorized")
        self.assertAlmostEqual(prediction.v, storage.data.back().output.v, 1, "Wrong value memorized")

        cdef TinyDnnModel other_model = TinyDnnModel()
        other_model.copy_weights_from(model)

        cdef ModelOutput other_prediction
        other_model.predict_single(&storage.data.back().obs, &other_prediction)
        for i in range(32):
            self.assertEqual(other_prediction.p[i], prediction.p[i], "Probabilities not equal")
        self.assertEqual(other_prediction.v, prediction.v, "Value not equal")
        """
