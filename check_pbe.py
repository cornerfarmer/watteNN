
from src.KerasModel import KerasModel
from src.Game import Game
from src.ModelRating import ModelRating
from gym_watten.envs.watten_env import WattenEnv

env = WattenEnv(True)
rating = ModelRating(env)
model = KerasModel(env, 128)
model.load('results/maskedSoftmax/minimal_env: True - sample_size: 3500/0/best-model')

game = Game(env)
table, avg_diff, max_diff = game.draw_game_tree(model, rating, False, 10)

print(table['-4,7,13,'])
print(avg_diff, max_diff)
