
from src.KerasModel import KerasModel
from src.Game import Game
from src.ModelRating import ModelRating
from gym_watten.envs.watten_env import WattenEnv

env = WattenEnv(True)
rating = ModelRating(env)
model = KerasModel(env, 128, clip=0.15)
model.load('results/true_loss/minimal_env: True - equalizer: 0 - batch_size: 128 - sample_size: 0 - value_lr: 0.01 - episodes: 100 - policy_lr: 0.01/4/best-model')

game = Game(env)
table, avg_diff, max_diff, v_loss = game.draw_game_tree(model, rating, False, 230, ',5.1-4,7,12,')#335,249,284

#print(table[',13-4,12,15,'])
print(avg_diff, max_diff)
print("v_loss: " + str(v_loss))
