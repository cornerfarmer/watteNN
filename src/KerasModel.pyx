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

import numpy as np
cimport numpy as np

import time

cdef extern from "<string>" namespace "std":
    string to_string(int val)

cdef class KerasModel(Model):
    def __init__(self):
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
        policy_out = Dense(256, activation='relu')(policy_out)
        policy_out = Dense(32, activation='sigmoid')(policy_out)

        value_out = concatenate([convnet, input_2])
        #value_out = Dense(64, activation='relu')(value_out)
        #value_out = Dense(128, activation='relu')(value_out)
        value_out = Dense(256, activation='relu')(value_out)
        value_out = Dense(1, activation='tanh')(value_out)

        self.model = RealKerasModel(inputs=[input_1, input_2], outputs=[policy_out, value_out])

        adam = optimizers.SGD(lr=0.1, momentum=0.9)
        self.model.compile(optimizer=adam,
                      loss='mean_squared_error',
                      metrics=['accuracy'])

    cpdef void memorize_storage(self, Storage storage, bool clear_afterwards=True, int epochs=1):
        cdef np.ndarray input1 = np.zeros([storage.data.size(), 4, 8, 6])
        cdef np.ndarray input2 = np.zeros([storage.data.size(), 4])

        cdef np.ndarray output1 = np.zeros([storage.data.size(), 32])
        cdef np.ndarray output2 = np.zeros([storage.data.size(), 1])

        for i in range(storage.data.size()):
            input1[i] = storage.data[i].obs.hand_cards
            input2[i] = storage.data[i].obs.tricks

            output1[i] = storage.data[i].output.p
            output2[i][0] = storage.data[i].output.v

        self.model.fit([input1, input2], [output1, output2], epochs=epochs, batch_size=64)

        if clear_afterwards:
            storage.data.clear()

    cdef void predict_single(self, Observation* obs, ModelOutput* output):
        cdef int i

        inputs = [np.array([obs.hand_cards]), np.array([obs.tricks])]

        outputs = self.model.predict(inputs)

        output.p = outputs[0][0]
        output.v = outputs[1][0][0]


    cdef void copy_weights_from(self, Model other_model):
        self.model.set_weights((<KerasModel>other_model).model.get_weights())