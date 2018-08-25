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
import time as pytime

class WatteNNTask(taskplan.Task):

    def __init__(self, preset, preset_pipe, logger):
        super().__init__(preset, preset_pipe, logger)
        self.sum = 0
        self.env = WattenEnv(self.preset.get_bool("minimal_env"))
        self.model = KerasModel(self.env, self.preset.get_int("hidden_neurons"), self.preset.get_int("batch_size"), self.preset.get_float("lr"), self.preset.get_float("momentum"), 0.15, self.preset.get_float("equalizer"))
        self.best_model = KerasModel(self.env, self.preset.get_int("hidden_neurons"), self.preset.get_int("batch_size"), self.preset.get_float("lr"), self.preset.get_float("momentum"), 0.15, self.preset.get_float("equalizer"))
        self.train_model = KerasModel(self.env, self.preset.get_int("hidden_neurons"), self.preset.get_int("batch_size"), self.preset.get_float("lr"), self.preset.get_float("momentum"), 0.15, self.preset.get_float("equalizer"))
        self.storage = Storage(self.preset.get_int("storage_size"))
        self.mcts = MCTS(self.preset.get_int("episodes"), self.preset.get_int("mcts_sims"), exploration=self.preset.get_float("exploration"))
        self.game = Game(self.env)
        self.train_model.copy_weights_from(self.model)
        self.best_model.copy_weights_from(self.model)
        self.rating = ModelRating(self.env)

    def save(self, path):
        self.best_model.save(str(path / Path('best-model')))
        self.train_model.save(str(path / Path('train-model')))
        self.model.save(str(path / Path('model')))

    def step(self, tensorboard_writer, current_iteration):
        start = pytime.time()
        self.mcts.mcts_generate(self.env, self.model, self.storage)
        print("1", pytime.time() - start)

        start = pytime.time()
        loss = self.train_model.memorize_storage(self.storage, self.preset.get_int('sample_size') != 0, self.preset.get_int('epochs'), self.preset.get_int('sample_size'))
        print("2", pytime.time() - start)

        tensorboard_writer.add_summary(tf.Summary(value=[
            tf.Summary.Value(tag="loss_play", simple_value=loss[0]),
            tf.Summary.Value(tag="loss_value", simple_value=loss[1]),
            tf.Summary.Value(tag="loss_choose", simple_value=loss[2])
        ]), current_iteration)

        self.model.copy_weights_from(self.train_model)
        if False and current_iteration % 1 == 0:
            self.model.copy_weights_from(self.train_model)

            rating_value = self.game.compare_rand_games(self.model, self.best_model, 500)
            tensorboard_writer.add_summary(tf.Summary(value=[
                tf.Summary.Value(tag="mean_game_length", simple_value=self.game.mean_game_length),
                tf.Summary.Value(tag="win_rate", simple_value=rating_value),
                tf.Summary.Value(tag="mean_v_p1", simple_value=self.game.mean_v_p1)
            ]), current_iteration)

            if rating_value > 0.52:
                self.best_model.copy_weights_from(self.model)

        if self.preset.get_bool("minimal_env") and current_iteration % self.preset.get_int("exploit_interval") == 0:
            self.best_model.copy_weights_from(self.model)
            table, avg_diff, max_diff = self.game.draw_game_tree(self.best_model, self.rating, False, None)

            tensorboard_writer.add_summary(tf.Summary(value=[
                tf.Summary.Value(tag="pbe", simple_value=avg_diff)
            ]), current_iteration)

    def load(self, path):
        self.best_model.load(str(path / Path('best-model')))
        self.train_model.load(str(path / Path('train-model')))
        self.model.load(str(path / Path('model')))