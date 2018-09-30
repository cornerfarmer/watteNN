
from src.KerasModel import KerasModel
from src.Game import Game
from src.ModelRating import ModelRating
from gym_watten.envs.watten_env import WattenEnv

env = WattenEnv(True)
rating = ModelRating(env)
model = KerasModel(env, 256, clip=0.15)
model.load('results/new_rand/hidden_neurons: 256 - episodes: 50 - mcts_sims: 100 - clear_samples_after_epoch: True/0/best-model')

game = Game(env)
table, avg_diff_on, avg_diff_off, max_diff, v_loss_on, v_loss_off, v_based_avg_diff_on, v_based_avg_diff_off = game.draw_game_tree(model, rating, False, 120, ',14.0,12.1-4,6,')#335,249,284

#print(table[',13-4,12,15,'])
print(avg_diff_on, avg_diff_off, max_diff)
print("v_loss on/off: " + str(v_loss_on) + ", " + str(v_loss_off))
