from libcpp.string cimport string
from libcpp cimport bool
from gym_watten.envs.watten_env cimport Observation, WattenEnv, Card
from src.MCTS cimport Storage
from src cimport ModelOutput
from src.Model cimport Model

from keras.models import Sequential, clone_model
from keras.layers import Dense, Activation, Input, LeakyReLU
from keras.layers import Conv2D, MaxPooling2D, Merge, Flatten, BatchNormalization, add
from keras.layers.merge import concatenate
from keras.models import Model as RealKerasModel
from keras.models import load_model
from keras import optimizers
from libc.stdlib cimport rand

import numpy as np
cimport numpy as np

import time

cdef extern from "<string>" namespace "std":
    string to_string(int val)

cdef class KerasModel(Model):
    def __init__(self, hidden_neurons=128):
        input_1 = Input((4,8,6))
        convnet = input_1

        #convnet = conv_layer(convnet, 75, (3, 3))
       # convnet = residual_layer(convnet, 75, (3, 3))
       # xonvnet = residual_layer(convnet, 75, (3, 3))
       # convnet = residual_layer(convnet, 75, (3, 3))
       # convnet = residual_layer(convnet, 75, (3, 3))
       # convnet = residual_layer(convnet, 75, (3, 3))
       # convnet = conv_layer(convnet, 2, (1, 1))
        #convnet = Conv2D(32, (3, 3), activation='relu', padding='same')(convnet)
        #convnet = Conv2D(64, (3, 3), activation='relu', padding='same')(convnet)
        #convnet = Conv2D(64, (3, 3), activation='relu', padding='same')(convnet)
        #convnet = Conv2D(64, (3, 3), activation='relu', padding='same')(convnet)
        convnet = Flatten()(convnet)

        input_2 = Input((4,))

        policy_out = concatenate([convnet, input_2])
        #policy_out = Dense(64, activation='relu')(policy_out)
        #policy_out = Dense(128, activation='relu')(policy_out)
        policy_out = Dense(hidden_neurons, activation='relu')(policy_out)
        policy_out = Dense(32, activation='sigmoid')(policy_out)

        value_out = concatenate([convnet, input_2])
        #value_out = Dense(64, activation='relu')(value_out)
        #value_out = Dense(128, activation='relu')(value_out)
        value_out = Dense(hidden_neurons, activation='relu')(value_out)
        value_out = Dense(1, activation='tanh')(value_out)

        self.model = RealKerasModel(inputs=[input_1, input_2], outputs=[policy_out, value_out])

       # adam = optimizers.SGD(lr=0.01, momentum=0)
        adam = optimizers.Adam()
        self.model.compile(optimizer=adam,
                      loss='mean_squared_error',
                      metrics=['accuracy'])
        self.clean_opt_weights = None

    cpdef void memorize_storage(self, Storage storage, bool clear_afterwards=True, int epochs=1, int number_of_samples=0):
        cdef bool use_random_selection = (number_of_samples is 0)
        number_of_samples = max(number_of_samples, storage.number_of_samples)

        cdef int s = storage.data.size() if use_random_selection else number_of_samples
        cdef np.ndarray input1 = np.zeros([s, 4, 8, 6])
        cdef np.ndarray input2 = np.zeros([s, 4])

        cdef np.ndarray output1 = np.zeros([s, 32])
        cdef np.ndarray output2 = np.zeros([s, 1])

        if self.clean_opt_weights is not None:
            self.model.optimizer.set_weights(self.clean_opt_weights)

        cdef int sample_index = 0
        for i in range(s):
            if use_random_selection:
                sample_index = i
            else:
                sample_index = rand() % storage.number_of_samples

            input1[i] = storage.data[sample_index].obs.hand_cards
            input2[i] = storage.data[sample_index].obs.tricks

            output1[i] = storage.data[sample_index].output.p
            output2[i][0] = storage.data[sample_index].output.v

        #print("Loss ", self.model.test_on_batch([input1, input2], [output1, output2]))
        self.model.fit([input1, input2], [output1, output2], epochs=epochs, batch_size=1)
        #print("Loss ", self.model.test_on_batch([input1, input2], [output1, output2]))
        #print(self.model.get_weights()[-4:-2])

        if self.clean_opt_weights is None:
            self.clean_opt_weights = self.model.optimizer.get_weights()
            for weight in self.clean_opt_weights:
                weight.fill(0)

        if clear_afterwards:
            storage.data.clear()

    cdef void predict_single(self, Observation* obs, ModelOutput* output):
        cdef int i

        inputs = [np.array([obs.hand_cards]), np.array([obs.tricks])]

        outputs = self.model.predict(inputs)

        output.p = outputs[0][0]
        output.v = outputs[1][0][0]


    cpdef void copy_weights_from(self, Model other_model):
        self.model.set_weights((<KerasModel>other_model).model.get_weights())