
from src.KerasModel import KerasModel
from src.ModelRating import ModelRating
from src.MCTS import MCTS
from gym_watten.envs.watten_env import WattenEnv

env = WattenEnv(True)
rating = ModelRating(env)
mcts = MCTS(mcts_sims=20)
model = KerasModel(env, 128)
model.load('results/onehot/minimal_env: True/0/best-model')

p = mcts.draw_game_tree(rating, env, model, 10, 5, [1])

print(p)

