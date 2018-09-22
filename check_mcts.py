
from src.KerasModel import KerasModel
from src.ModelRating import ModelRating
from src.MCTS import MCTS
from src.Storage import Storage
from gym_watten.envs.watten_env import WattenEnv

env = WattenEnv(True)
rating = ModelRating(env)
mcts = MCTS(episodes=1, mcts_sims=40, exploration=0.1, only_one_step=True)
model = KerasModel(env, 128, clip=0.15)
model.load('results/new_w/minimal_env: True - equalizer: 0 - batch_size: 128 - sample_size: 0 - value_lr: 0.01 - episodes: 100 - policy_lr: 0.01/0/best-model')

storage = Storage()
mcts.draw_game_tree(rating, env, model, storage, 230, 5, [0,1])#495,185



