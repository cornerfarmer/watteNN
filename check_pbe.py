
from src.KerasModel import KerasModel
from src.Game import Game
from src.ModelRating import ModelRating
from gym_watten.envs.watten_env import WattenEnv

env = WattenEnv(True)
rating = ModelRating(env)
model = KerasModel(env, 128)
model.load('results/dualWeighting/batch_size: 128 - lr: 0.1 - minimal_env: True - sample_size: 0 - storage_size: 80000/0/best-model')

game = Game(env)
table, avg_diff, max_diff = game.draw_game_tree(model, rating, False, 335)#374

print(table[',13-4,12,15,'])
print(avg_diff, max_diff)
