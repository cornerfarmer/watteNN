
from src.KerasModel import KerasModel
from src.Game import Game
from src.ModelRating import ModelRating
from gym_watten.envs.watten_env import WattenEnv

env = WattenEnv(True)
rating = ModelRating(env)
model = KerasModel(env, 128, clip=2)
model.load('results/momentumFix/batch_size: 128 - clip: 0.85 - episodes: 100 - lr: 0.1 - minimal_env: True - sample_size: 0/2/best-model')

game = Game(env)
table, avg_diff, max_diff = game.draw_game_tree(model, rating, False, 335)#335,249

print(table[',13-4,12,15,'])
print(avg_diff, max_diff)
