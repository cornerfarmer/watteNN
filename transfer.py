
from src.KerasModel import KerasModel
from src.ModelRating import ModelRating
from src.Storage import Storage
from src.LookUp import LookUp
from src.MCTS import MCTS
from gym_watten.envs.watten_env import WattenEnv
import pickle
from src.XorShfGenerator import XorShfGenerator
from src.Game import Game

env = WattenEnv(True)
rating = ModelRating(env)
rng = XorShfGenerator()
#model =
model = LookUp(clip=0)
storage = Storage()
path = 'results/lookup/clear_samples_after_epoch: True - episodes: 50 - mcts_sims: 100 - policy_lr: 0.01 - policy_momentum: 0.99 - step_exploration: 0.9 - value_lr: 0.005/0'
#path += '/checkpoints/2800'
model.load(path + '/best-model')

model.generate_storage(storage, env)

network = KerasModel(env, 128, equalizer=0, clip=0.15, batch_size=128, policy_lr=0.01, policy_momentum=0.8, value_lr=0.005, value_momentum=0.9)
network.memorize_storage(storage, False, 3000, 0)



game = Game(env)
table, avg_diff_on, avg_diff_off, max_diff, v_loss_on, v_loss_off, v_based_avg_diff_on, v_based_avg_diff_off = game.draw_game_tree(network, rating, False, 0, None)#335,249,284

print(avg_diff_on, avg_diff_off, max_diff)
print("v_loss on/off: " + str(v_loss_on) + ", " + str(v_loss_off))