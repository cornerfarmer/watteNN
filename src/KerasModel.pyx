from libcpp.string cimport string
from libcpp cimport bool
from gym_watten.envs.watten_env cimport Observation, WattenEnv, Card, ActionType
from src.MCTS cimport Storage
from src cimport ModelOutput
from src.Model cimport Model

from keras.models import Sequential, clone_model
from keras.layers import Dense, Activation, Input, LeakyReLU
from keras.layers import Conv2D, MaxPooling2D, Merge, Flatten, BatchNormalization, add, Multiply, Lambda, Activation
from keras.layers.merge import concatenate
from keras.models import Model as RealKerasModel
from keras.models import load_model
from keras import optimizers
from keras.losses import mean_absolute_error
import keras.backend as K
from libc.stdlib cimport rand
from keras.engine.topology import Layer

import numpy as np
cimport numpy as np

import time

cdef extern from "<string>" namespace "std":
    string to_string(int val)

class SelectiveSoftmax(Layer):
    def __init__(self, **kwargs):
        super(SelectiveSoftmax, self).__init__(**kwargs)

    def build(self, input_shape):
        super(SelectiveSoftmax, self).build(input_shape)

    def call(self, inputs):
        exp = K.exp(inputs[0] - K.max(inputs[0], -1, True))
        exp *= inputs[1]
        return exp / K.sum(exp, -1, True)

    def compute_output_shape(self, input_shape):
        return input_shape[0]

cdef class KerasModel(Model):
    def __init__(self, env, hidden_neurons=128, batch_size=30, lr=0.04, momentum=0.9, clip=2, equalizer=0.01):
        self.lr = lr
        self.momentum = momentum

        self._build_choose_model(env, hidden_neurons)
        self._build_play_model(env, hidden_neurons)
        self._build_value_model(env, hidden_neurons)

        self.clean_opt_weights = None
        self.batch_size = batch_size
        self.clip = clip
        self.equalizer = equalizer

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

        adam = optimizers.SGD(lr=self.lr, momentum=self.momentum)
        #adam = optimizers.Adam()
        self.choose_model.compile(optimizer=adam,
                      loss='mean_absolute_error',
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
        policy_out = Dense(32, activation='linear')(policy_out)
        def slice(x):
            return x[:, :, :, 0]
        input_slice = Lambda(slice)(input_1)
        input_slice = Flatten()(input_slice)
        policy_out = SelectiveSoftmax()([policy_out, input_slice])

        scale_out = concatenate([convnet, input_2])
        scale_out = Dense(hidden_neurons, activation='relu')(scale_out)
        scale_out = Dense(1, activation='elu')(scale_out)
        def add_one(x):
            return x + 1
        scale_out = Lambda(add_one)(scale_out)

        self.play_model = RealKerasModel(inputs=[input_1, input_2], outputs=[policy_out, scale_out])

        def customLoss(yTrue, yPred):
            a = 0.1
            loss_sq = 0.5 * 1 / a * K.square(yPred - yTrue)
            loss_abs = K.abs(yPred - yTrue) - 0.5 * 1 / a * a ** 2
            use_abs = K.abs(yPred - yTrue) > a
            loss = K.mean(K.cast(use_abs, 'float32') * loss_abs + (1 - K.cast(use_abs, 'float32')) * loss_sq, axis=-1)
            return loss + self.equalizer * K.max(yPred, axis=-1)

        adam = optimizers.SGD(lr=self.lr, momentum=self.momentum)
        #adam = optimizers.Adam()
        self.play_model.compile(optimizer=adam,
                      loss=[customLoss, 'mean_squared_error'],
                      metrics=['accuracy'])

    cdef void _build_value_model(self, WattenEnv env, int hidden_neurons):
        input_1 = Input((4, 8, 3))
        convnet = input_1
        convnet = Flatten()(convnet)

        input_2 = Input((4,))

        value_out = concatenate([convnet, input_2])
        value_out = Dense(hidden_neurons, activation='relu')(value_out)
        value_out = Dense(1, activation='tanh')(value_out)

        self.value_model = RealKerasModel(inputs=[input_1, input_2], outputs=[value_out])

        adam = optimizers.SGD(lr=self.lr, momentum=self.momentum)
        #adam = optimizers.Adam()
        self.value_model.compile(optimizer=adam,
                      loss=['mean_squared_error'],
                      metrics=['accuracy'])

    cpdef vector[float] memorize_storage(self, Storage storage, bool clear_afterwards=True, int epochs=1, int number_of_samples=0):
        cdef bool use_random_selection = (number_of_samples is 0)
        number_of_samples = min(number_of_samples, storage.number_of_samples)

        cdef int s = storage.number_of_samples if use_random_selection else number_of_samples
        print(s, storage.number_of_samples)
        cdef np.ndarray play_input1 = np.zeros([s, 4, 8, self.play_input_sets_size])
        cdef np.ndarray play_input2 = np.zeros([s, 4])

        cdef np.ndarray play_output1 = np.zeros([s, 32])
        cdef np.ndarray play_output2 = np.zeros([s, 1])

        play_weights = [np.zeros([s])]


        cdef np.ndarray value_input1 = np.zeros([s, 4, 8, 3])
        cdef np.ndarray value_input2 = np.zeros([s, 4])

        cdef np.ndarray value_output1 = np.zeros([s, 1])


        cdef np.ndarray choose_input1 = np.zeros([s, 4, 8, self.choose_input_sets_size])

        cdef np.ndarray choose_output1 = np.zeros([s, 32])
        cdef np.ndarray choose_output2 = np.zeros([s, 1])

        #if self.clean_opt_weights is not None:
        #    self.model.optimizer.set_weights(self.clean_opt_weights)

        cdef int play_index = 0, choose_index = 0, sample_index = 0, value_index = 0
        for i in range(s):
            if use_random_selection:
                sample_index = rand() % storage.number_of_samples
            else:
                sample_index = i


            if storage.data[sample_index].value_net:
                value_input1[value_index] = storage.data[sample_index].obs.sets
                value_input2[value_index] = storage.data[sample_index].obs.scalars

                value_output1[value_index][0] = storage.data[sample_index].output.v

                value_index += 1
            else:
                if storage.data[sample_index].obs.type is ActionType.DRAW_CARD:
                    play_input1[play_index] = storage.data[sample_index].obs.sets
                    play_input2[play_index] = storage.data[sample_index].obs.scalars
                    play_output1[play_index] = storage.data[sample_index].output.p
                    play_output2[play_index] = storage.data[sample_index].output.scale

                    play_weights[0][play_index] = storage.data[sample_index].weight
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
        play_weights[0].resize([play_index])

        value_input1.resize([value_index, 4, 8, 3])
        value_input2.resize([value_index, 4])
        value_output1.resize([value_index, 1])

        choose_input1.resize([choose_index, 4, 8, self.choose_input_sets_size])
        choose_output1.resize([choose_index, 32])
        choose_output2.resize([choose_index, 1])

        #print("Loss ", self.model.test_on_batch([input1, input2], [output1, output2]))
        cdef vector[float] loss
        loss.push_back(self.play_model.fit([play_input1, play_input2], [play_output1, play_output2], epochs=epochs, batch_size=min(play_index, self.batch_size), sample_weight=[play_weights[0], np.ones_like(play_weights[0])]).history['loss'][-1])
        loss.push_back(self.value_model.fit([value_input1, value_input2], [value_output1], epochs=epochs, batch_size=min(value_index, self.batch_size)).history['loss'][-1])
        if choose_index > 0:
            loss.push_back(self.choose_model.fit([choose_input1], [choose_output1, choose_output2], epochs=epochs, batch_size=min(choose_index, self.batch_size)).history['loss'][-1])
        else:
            loss.push_back(0)
        #print("Loss ", self.model.test_on_batch([input1, input2], [output1, output2]))
        #print(self.model.get_weights()[-4:-2])

        #if self.clean_opt_weights is None:
        #    self.clean_opt_weights = self.model.optimizer.get_weights()
        #    for weight in self.clean_opt_weights:
        #        weight.fill(0)

        if clear_afterwards:
            storage.clear()
        return loss


    cdef void predict_single_p(self, Observation* obs, ModelOutput* output):
        cdef int i

        if obs.type is ActionType.DRAW_CARD:
            inputs = [np.array([obs.sets]), np.array([obs.scalars])]
            outputs = self.play_model.predict(inputs)
        else:
            inputs = [np.array([obs.sets])]
            outputs = self.choose_model.predict(inputs)

        if np.max(outputs[0][0]) >= self.clip:
            output.p = (outputs[0][0] >= self.clip)
        else:
            output.p = outputs[0][0]
        output.scale = outputs[1][0]

    cdef float predict_single_v(self, Observation* full_obs):
        inputs = [np.array([full_obs.sets]), np.array([full_obs.scalars])]
        outputs = self.value_model.predict(inputs)
        return outputs[0][0]


    cdef void predict_p(self, vector[Observation]* obs, vector[ModelOutput]* output):
        cdef int i

        #if obs.type is ActionType.DRAW_CARD:
        inputs = [np.zeros([obs.size(), 4, 8, self.play_input_sets_size]), np.zeros([obs.size(), 4])]
        for i in range(obs.size()):
            inputs[0][i] = obs[0][i].sets
            inputs[1][i] = obs[0][i].scalars
        outputs = self.play_model.predict(inputs)
        #else:
        #    inputs = [np.array([obs.sets])]
        #    outputs = self.choose_model.predict(inputs)

        for i in range(output.size()):
            if np.max(outputs[0][i]) >= self.clip:
                output[0][i].p = (outputs[0][i] >= self.clip)
            else:
                output[0][i].p = outputs[0][i]
            output[0][i].scale = outputs[1][i]

    cdef void predict_v(self, vector[Observation]* full_obs, vector[ModelOutput]* output):
        inputs = [np.zeros([full_obs.size(), 4, 8, 3]), np.zeros([full_obs.size(), 4])]
        for i in range(full_obs.size()):
            inputs[0][i] = full_obs[0][i].sets
            inputs[1][i] = full_obs[0][i].scalars
        outputs = self.value_model.predict(inputs)

        for i in range(output.size()):
            output[0][i].v = outputs[i]

    cpdef void copy_weights_from(self, Model other_model):
        self.choose_model.set_weights((<KerasModel>other_model).choose_model.get_weights())
        self.play_model.set_weights((<KerasModel>other_model).play_model.get_weights())
        self.value_model.set_weights((<KerasModel>other_model).value_model.get_weights())

    cpdef void load(self, filename):
        self.play_model.load_weights(filename + "-play")
        self.choose_model.load_weights(filename + "-choose")
        self.value_model.load_weights(filename + "-value")

    cpdef void save(self, filename):
        self.play_model.save(filename + "-play")
        self.choose_model.save(filename + "-choose")
        self.value_model.save(filename + "-value")