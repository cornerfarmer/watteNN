
from src.KerasModel import KerasModel
from src.ModelRating import ModelRating
from src.MCTS import MCTS
from src.Storage import Storage
from gym_watten.envs.watten_env import WattenEnv

from src.XorShfGenerator import XorShfGenerator

env = WattenEnv(True)
rating = ModelRating(env)
rng = XorShfGenerator()
mcts = MCTS(rng, episodes=1, mcts_sims=100, exploration=0.1, only_one_step=True)
model = KerasModel(env, 128, clip=0.15)
model.load('results/remote/mcts_sims: 100/0/best-model')

storage = Storage()
mcts.draw_game_tree(rating, env, model, storage, 93, 5, [2, 1])#495,185



