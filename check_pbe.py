
from src.KerasModel import KerasModel
from src.Game import Game
from src.ModelRating import ModelRating
from gym_watten.envs.watten_env import WattenEnv

env = WattenEnv(True)
rating = ModelRating(env)
model = KerasModel(env, 128, clip=0.15)
model.load('results/momentumFix/batch_size: 128 - episodes: 100 - equalizer: 0 - minimal_env: True - policy_lr: 0.01 - sample_size: 0 - value_lr: 0.01/0/best-model')

game = Game(env)
table, avg_diff, max_diff = game.draw_game_tree(model, rating, False, 230, ',5.1-4,7,12,')#335,249,284

#print(table[',13-4,12,15,'])
print(avg_diff, max_diff)
