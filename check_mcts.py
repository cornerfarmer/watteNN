
from src.KerasModel import KerasModel
from src.ModelRating import ModelRating
from src.MCTS import MCTS
from src.Storage import Storage
from gym_watten.envs.watten_env import WattenEnv

env = WattenEnv(True)
rating = ModelRating(env)
mcts = MCTS(episodes=1, mcts_sims=40, exploration=0.1, only_one_step=True)
model = KerasModel(env, 256, clip=0.15)
model.load('results/val_rec/remote/1/best-model')

storage = Storage()
mcts.draw_game_tree(rating, env, model, storage, 335, 5, [1])#495,185



