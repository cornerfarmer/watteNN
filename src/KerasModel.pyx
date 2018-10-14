from keras.callbacks import LambdaCallback
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
        masked = K.tf.where(K.tf.greater(inputs[1], inputs[1] * 0), inputs[0], K.tf.ones_like(inputs[0]) * K.constant(-np.inf))
        exp = K.exp(inputs[0] - K.max(masked, -1, True))
        exp *= inputs[1]
        return exp / K.sum(exp, -1, True)

    def compute_output_shape(self, input_shape):
        return input_shape[0]

cdef class KerasModel(Model):
    def __init__(self, env, hidden_neurons=128, batch_size=30, policy_lr=0.04, policy_momentum=0.9, value_lr=0.04, value_momentum=0.9, clip=0, equalizer=0.01):
        self.policy_lr = policy_lr
        self.policy_momentum = policy_momentum
        self.value_lr = value_lr
        self.value_momentum = value_momentum
        self.clean_opt_weights = None
        self.batch_size = batch_size
        self.clip = clip
        self.equalizer = equalizer

        self._build_choose_model(env, hidden_neurons)
        self._build_play_model(env, hidden_neurons)
        self._build_value_model(env, hidden_neurons)

        self.test_obs = {'sets': [[[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0]], [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]]], 'scalars': [1, 0, 0, 0, 0, 1, 0, 0], 'type': ActionType.DRAW_CARD}
        self.test_obs_p = {'sets': [[[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]], 'scalars': [1, 0, 0, 0, 0, 1, 0, 0], 'type': ActionType.DRAW_CARD}


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

        adam = optimizers.SGD(lr=self.policy_lr, momentum=self.policy_momentum)
        #adam = optimizers.Adam()
        self.choose_model.compile(optimizer=adam,
                      loss='mean_absolute_error',
                      metrics=['accuracy'])

    cdef void _build_play_model(self, WattenEnv env, int hidden_neurons):
        self.play_input_sets_size = env.get_input_sets_size(ActionType.DRAW_CARD)
        self.play_input_scalars_size = env.get_input_scalars_size(ActionType.DRAW_CARD)
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

        input_2 = Input((self.play_input_scalars_size,))

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


        self.play_model = RealKerasModel(inputs=[input_1, input_2], outputs=[policy_out])

        def customLoss(yTrue, yPred):
            a = 0.1
            loss_sq = 0.5 * 1 / a * K.pow(yPred - yTrue, 10)
            loss_abs = K.abs(yPred - yTrue) - 0.5 * 1 / a * a ** 10
            use_abs = K.abs(yPred - yTrue) > a
            loss = K.mean(K.cast(use_abs, 'float32') * loss_abs + (1 - K.cast(use_abs, 'float32')) * loss_sq, axis=-1)
            return loss

        opt = optimizers.SGD(lr=self.policy_lr, momentum=self.policy_momentum)
        self.play_model.compile(optimizer=opt,
                      loss=[customLoss],
                      metrics=['accuracy'])

    cdef void _build_value_model(self, WattenEnv env, int hidden_neurons):
        self.play_input_sets_size = env.get_input_sets_size(ActionType.DRAW_CARD)
        self.play_input_scalars_size = env.get_input_scalars_size(ActionType.DRAW_CARD)
        input_1 = Input((4, 8, self.play_input_sets_size + 1))
        convnet = input_1
        convnet = Flatten()(convnet)

        input_2 = Input((self.play_input_scalars_size,))

        value_out = concatenate([convnet, input_2])
        value_out = Dense(hidden_neurons, activation='relu')(value_out)
        value_out = Dense(1, activation='linear')(value_out)

        self.value_model = RealKerasModel(inputs=[input_1, input_2], outputs=[value_out])

        opt = optimizers.SGD(lr=self.value_lr, momentum=self.value_momentum)
        #adam = optimizers.Adam()
        self.value_model.compile(optimizer=opt,
                      loss=['mean_squared_error'],
                      metrics=['accuracy'])
    cpdef predict_v_model(self, epoch, logs):
        print("\nv: " + str(self.predict_single_v(&self.test_obs)))


    cpdef predict_p_model(self, epoch, logs):
        cdef ModelOutput output
        self.predict_single_p(&self.test_obs_p, &output)
        print("\np: " + str(output.p))

    cpdef vector[float] memorize_storage(self, Storage storage, bool clear_afterwards=True, int epochs=1, int number_of_samples=0):
        if number_of_samples == 0:
            number_of_samples = storage.number_of_samples

        number_of_samples = min(number_of_samples, storage.number_of_samples)

        cdef int s = number_of_samples

        cdef np.ndarray play_input1 = np.zeros([s, 4, 8, self.play_input_sets_size])
        cdef np.ndarray play_input2 = np.zeros([s, self.play_input_scalars_size])

        cdef np.ndarray play_output1 = np.zeros([s, 32])

        play_weights = [np.zeros([s])]


        cdef np.ndarray value_input1 = np.zeros([s, 4, 8, self.play_input_sets_size + 1])
        cdef np.ndarray value_input2 = np.zeros([s, self.play_input_scalars_size])

        cdef np.ndarray value_output1 = np.zeros([s, 1])


        cdef np.ndarray choose_input1 = np.zeros([s, 4, 8, self.choose_input_sets_size])

        cdef np.ndarray choose_output1 = np.zeros([s, 32])
        cdef np.ndarray choose_output2 = np.zeros([s, 1])

        #if self.clean_opt_weights is not None:
        #    self.model.optimizer.set_weights(self.clean_opt_weights)

        cdef int play_index = 0, choose_index = 0, sample_index = 0, value_index = 0
        for i in range(s):
            if number_of_samples < storage.number_of_samples:
                sample_index = rand() % storage.number_of_samples
            else:
                sample_index = i


            if storage.data[sample_index].value_net:
                if storage.data[sample_index].obs.sets[0][5][0] == 1 and storage.data[sample_index].obs.sets[1][5][0] == 1 and storage.data[sample_index].obs.sets[1][4][1] == 1 and storage.data[sample_index].obs.sets[1][6][1] == 1 and storage.data[sample_index].obs.sets[0][7][3] == 1 and storage.data[sample_index].obs.sets[1][7][4] == 1:
                    print("learn", storage.data[sample_index].output.v)
                value_input1[value_index] = storage.data[sample_index].obs.sets
                value_input2[value_index] = storage.data[sample_index].obs.scalars

                value_output1[value_index][0] = storage.data[sample_index].output.v

                value_index += 1
            else:
                if storage.data[sample_index].obs.type is ActionType.DRAW_CARD:
                    play_input1[play_index] = storage.data[sample_index].obs.sets
                    play_input2[play_index] = storage.data[sample_index].obs.scalars
                    play_output1[play_index] = storage.data[sample_index].output.p

                    play_weights[0][play_index] = storage.data[sample_index].weight
                    play_index += 1

                else:
                    choose_input1[choose_index] = storage.data[sample_index].obs.sets

                    choose_output1[choose_index] = storage.data[sample_index].output.p
                    choose_output2[choose_index][0] = storage.data[sample_index].output.v
                    choose_index += 1

        play_input1.resize([play_index, 4, 8, self.play_input_sets_size])
        play_input2.resize([play_index, self.play_input_scalars_size])
        play_output1.resize([play_index, 32])
        play_weights[0].resize([play_index])

        value_input1.resize([value_index, 4, 8, self.play_input_sets_size + 1])
        value_input2.resize([value_index, self.play_input_scalars_size])
        value_output1.resize([value_index, 1])

        choose_input1.resize([choose_index, 4, 8, self.choose_input_sets_size])
        choose_output1.resize([choose_index, 32])
        choose_output2.resize([choose_index, 1])

        print(play_index)


        # Callback to display the target and prediciton
        callback_v = LambdaCallback(on_epoch_end=self.predict_v_model)
        callback_p = LambdaCallback(on_epoch_end=self.predict_p_model)

        #print("Loss ", self.model.test_on_batch([input1, input2], [output1, output2]))
        cdef vector[float] loss
        loss.push_back(self.play_model.fit([play_input1, play_input2], [play_output1], epochs=epochs, batch_size=min(play_index, self.batch_size), sample_weight=[play_weights[0]], verbose=False, callbacks=[callback_p]).history['loss'][-1])
        #loss.push_back(0)
        loss.push_back(self.value_model.fit([value_input1, value_input2], [value_output1], epochs=epochs, batch_size=min(value_index, self.batch_size), callbacks=[]).history['loss'][-1])
        if choose_index > 0:
            loss.push_back(self.choose_model.fit([choose_input1], [choose_output1, choose_output2], epochs=epochs, batch_size=min(choose_index, self.batch_size)).history['loss'][-1])
        else:
            loss.push_back(0)
        #print("Loss ", self.model.test_on_batch([input1, input2], [output1, output2]))
        #print(self.model.get_weights()[-4:-2])
        print(loss)
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

        self._clip_output(outputs)
        output.p = outputs[0]

    cdef float predict_single_v(self, Observation* full_obs):
        inputs = [np.array([full_obs.sets]), np.array([full_obs.scalars])]
        outputs = self.value_model.predict(inputs)
        return np.clip(outputs[0][0], -1, 1)


    cdef void predict_p(self, vector[Observation]* obs, vector[ModelOutput]* output):
        cdef int i

        #if obs.type is ActionType.DRAW_CARD:
        inputs = [np.zeros([obs.size(), 4, 8, self.play_input_sets_size]), np.zeros([obs.size(), self.play_input_scalars_size])]
        for i in range(obs.size()):
            inputs[0][i] = obs[0][i].sets
            inputs[1][i] = obs[0][i].scalars
        outputs = self.play_model.predict(inputs)
        #else:
        #    inputs = [np.array([obs.sets])]
        #    outputs = self.choose_model.predict(inputs)
        self._clip_output(outputs)
        for i in range(output.size()):
            output[0][i].p = outputs[i]

    cdef object _clip_output(self, output):
        col_sum = output.copy()
        n = np.sum(col_sum >= self.clip, axis=-1)
        col_sum[col_sum >= self.clip] = 0
        col_sum = np.sum(col_sum, axis=-1)

        output[output < self.clip] = 0
        output[output >= self.clip] += np.repeat(np.expand_dims(1 / n * col_sum, axis=-1), output.shape[1], axis=1)[output >= self.clip]
        return output

    cdef void predict_v(self, vector[Observation]* full_obs, vector[ModelOutput]* output):
        inputs = [np.zeros([full_obs.size(), 4, 8, self.play_input_sets_size + 1]), np.zeros([full_obs.size(), self.play_input_scalars_size])]
        for i in range(full_obs.size()):
            inputs[0][i] = full_obs[0][i].sets
            inputs[1][i] = full_obs[0][i].scalars
        outputs = self.value_model.predict(inputs)

        for i in range(output.size()):
            output[0][i].v = np.clip(outputs[i], -1, 1)

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
