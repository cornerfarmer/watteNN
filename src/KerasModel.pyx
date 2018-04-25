from libcpp.string cimport string
from libcpp cimport bool
from gym_watten.envs.watten_env cimport Observation, WattenEnv, Card, ActionType
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
    def __init__(self, env, hidden_neurons=128):

        self._build_choose_model(env, hidden_neurons)
        self._build_play_model(env, hidden_neurons)

        self.clean_opt_weights = None

    cdef void _build_choose_model(self, WattenEnv env, int hidden_neurons):
        self.choose_input_sets_size = env.get_input_sets_size(ActionType.CHOOSE_VALUE)
        input_1 = Input((4,8, self.choose_input_sets_size))
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

        #policy_out = Dense(64, activation='relu')(policy_out)
        #policy_out = Dense(128, activation='relu')(policy_out)
        policy_out = Dense(hidden_neurons, activation='relu')(convnet)
        policy_out = Dense(32, activation='sigmoid')(policy_out)

        #value_out = Dense(64, activation='relu')(value_out)
        #value_out = Dense(128, activation='relu')(value_out)
        value_out = Dense(hidden_neurons, activation='relu')(convnet)
        value_out = Dense(1, activation='tanh')(value_out)

        self.choose_model = RealKerasModel(inputs=[input_1], outputs=[policy_out, value_out])

        adam = optimizers.SGD(lr=0.04, momentum=0.9)
        #adam = optimizers.Adam()
        self.choose_model.compile(optimizer=adam,
                      loss='mean_squared_error',
                      metrics=['accuracy'])

    cdef void _build_play_model(self, WattenEnv env, int hidden_neurons):
        self.play_input_sets_size = env.get_input_sets_size(ActionType.DRAW_CARD)
        input_1 = Input((4,8, self.play_input_sets_size))
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

        self.play_model = RealKerasModel(inputs=[input_1, input_2], outputs=[policy_out, value_out])

        adam = optimizers.SGD(lr=0.04, momentum=0.9)
        #adam = optimizers.Adam()
        self.play_model.compile(optimizer=adam,
                      loss='mean_squared_error',
                      metrics=['accuracy'])

    cpdef vector[float] memorize_storage(self, Storage storage, bool clear_afterwards=True, int epochs=1, int number_of_samples=0):
        cdef bool use_random_selection = (number_of_samples is 0)
        number_of_samples = min(number_of_samples, storage.number_of_samples)

        cdef int s = storage.data.size() if use_random_selection else number_of_samples
        cdef np.ndarray play_input1 = np.zeros([s, 4, 8, self.play_input_sets_size])
        cdef np.ndarray play_input2 = np.zeros([s, 4])

        cdef np.ndarray play_output1 = np.zeros([s, 32])
        cdef np.ndarray play_output2 = np.zeros([s, 1])


        cdef np.ndarray choose_input1 = np.zeros([s, 4, 8, self.choose_input_sets_size])

        cdef np.ndarray choose_output1 = np.zeros([s, 32])
        cdef np.ndarray choose_output2 = np.zeros([s, 1])


        #if self.clean_opt_weights is not None:
        #    self.model.optimizer.set_weights(self.clean_opt_weights)

        cdef int play_index = 0, choose_index = 0, sample_index = 0
        for i in range(s):
            if use_random_selection:
                sample_index = i
            else:
                sample_index = rand() % storage.number_of_samples

            if storage.data[sample_index].obs.type is ActionType.DRAW_CARD:
                play_input1[play_index] = storage.data[sample_index].obs.sets
                play_input2[play_index] = storage.data[sample_index].obs.scalars

                play_output1[play_index] = storage.data[sample_index].output.p
                play_output2[play_index][0] = storage.data[sample_index].output.v
                play_index += 1
            else:
                choose_input1[choose_index] = storage.data[sample_index].obs.sets

                choose_output1[choose_index] = storage.data[sample_index].output.p
                choose_output2[choose_index][0] = storage.data[sample_index].output.v
                choose_index += 1

        play_input1.resize([play_index, 4, 8, self.play_input_sets_size])
        play_input2.resize([play_index, 4])
        play_output1.resize([play_index, 32])
        play_output2.resize([play_index, 1])

        choose_input1.resize([choose_index, 4, 8, self.choose_input_sets_size])
        choose_output1.resize([choose_index, 32])
        choose_output2.resize([choose_index, 1])


        #print("Loss ", self.model.test_on_batch([input1, input2], [output1, output2]))
        cdef vector[float] loss
        loss.push_back(self.play_model.fit([play_input1, play_input2], [play_output1, play_output2], epochs=epochs, batch_size=min(play_index, 8)).history['loss'][-1])
        if choose_index > 0:
            loss.push_back(self.choose_model.fit([choose_input1], [choose_output1, choose_output2], epochs=epochs, batch_size=min(choose_index, 8)).history['loss'][-1])
        else:
            loss.push_back(0)
        #print("Loss ", self.model.test_on_batch([input1, input2], [output1, output2]))
        #print(self.model.get_weights()[-4:-2])

        #if self.clean_opt_weights is None:
        #    self.clean_opt_weights = self.model.optimizer.get_weights()
        #    for weight in self.clean_opt_weights:
        #        weight.fill(0)

        if clear_afterwards:
            storage.data.clear()
        return loss

    cdef void predict_single(self, Observation* obs, ModelOutput* output):
        cdef int i

        if obs.type is ActionType.DRAW_CARD:
            inputs = [np.array([obs.sets]), np.array([obs.scalars])]
            outputs = self.play_model.predict(inputs)
        else:
            inputs = [np.array([obs.sets])]
            outputs = self.choose_model.predict(inputs)

        output.p = outputs[0][0]
        output.v = outputs[1][0][0]

    cpdef void copy_weights_from(self, Model other_model):
        self.choose_model.set_weights((<KerasModel>other_model).choose_model.get_weights())
        self.play_model.set_weights((<KerasModel>other_model).play_model.get_weights())

    cpdef void load(self, filename):
        self.play_model.load_weights("play-" + filename)
        self.choose_model.load_weights("choose-" + filename)

    cpdef void save(self, filename):
        self.play_model.save("play-" + filename)
        self.choose_model.save("choose-" + filename)