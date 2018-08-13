
from src.KerasModel import KerasModel
from src.Game import Game
from src.ModelRating import ModelRating
from gym_watten.envs.watten_env import WattenEnv

env = WattenEnv(True)
rating = ModelRating(env)
model = KerasModel(env, 128)
model.load('results/dualWeighting/minimal_env: True - sample_size: 3500/0/best-model')

game = Game(env)
table = game.draw_game_tree(model, rating, False, 10)

print(table[',4,12,5-6,13,'])

