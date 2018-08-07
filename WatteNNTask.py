import pickle
from pathlib import Path

import tensorflow as tf
import time

import taskplan

from src.Storage import Storage
from src.MCTS import MCTS
from src.LookUp import LookUp
from src.KerasModel import KerasModel
from src.Game import Game
from src.ModelRating import ModelRating
from gym_watten.envs.watten_env import WattenEnv

class WatteNNTask(taskplan.Task):

    def __init__(self, preset, preset_pipe, logger):
        super().__init__(preset, preset_pipe, logger)
        self.sum = 0
        self.env = WattenEnv()
        self.model = KerasModel(self.env, self.preset.get_int("hidden_neurons"))
        self.best_model = KerasModel(self.env, self.preset.get_int("hidden_neurons"))
        self.train_model = KerasModel(self.env, self.preset.get_int("hidden_neurons"))
        self.storage = Storage(self.preset.get_int("storage_size"))
        self.mcts = MCTS(self.preset.get_int("episodes"), self.preset.get_int("mcts_sims"), self.preset.get_bool("objective_opponent"))
        self.game = Game(self.env)
        self.train_model.copy_weights_from(self.model)
        self.best_model.copy_weights_from(self.model)

    def save(self, path):
        self.best_model.save('best-model')
        self.train_model.save('train-model')
        self.model.save('model')

    def step(self, tensorboard_writer, current_iteration):
        self.mcts.mcts_generate(self.env, self.model, self.storage)

        loss = self.train_model.memorize_storage(self.storage, self.preset.get_int('sample_size') == 0, 1, self.preset.get_int('sample_size'))

        tensorboard_writer.add_summary(tf.Summary(value=[
            tf.Summary.Value(tag="loss_play", simple_value=loss[0]),
            tf.Summary.Value(tag="loss_choose", simple_value=loss[1])
        ]), current_iteration)

        if current_iteration % 1 == 0:
            self.model.copy_weights_from(self.train_model)

            rating_value = self.game.compare_rand_games(self.model, self.best_model, 500)
            tensorboard_writer.add_summary(tf.Summary(value=[
                tf.Summary.Value(tag="mean_game_length", simple_value=self.game.mean_game_length),
                tf.Summary.Value(tag="win_rate", simple_value=rating_value),
                tf.Summary.Value(tag="mean_v_p1", simple_value=self.game.mean_v_p1)
            ]), current_iteration)

            if rating_value > 0.52:
                self.best_model.copy_weights_from(self.model)


    def load(self, path):
        self.best_model.load('best-model')
        self.train_model.load('train-model')
        self.model.load('model')