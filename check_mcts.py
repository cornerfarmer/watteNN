
from src.KerasModel import KerasModel
from src.ModelRating import ModelRating
from src.MCTS import MCTS
from src.Storage import Storage
from gym_watten.envs.watten_env import WattenEnv

env = WattenEnv(True)
rating = ModelRating(env)
mcts = MCTS(episodes=1, mcts_sims=80, exploration=0.5, only_one_step=True)
model = KerasModel(env, 128, clip=0.15)
model.load('results/full_obs_fix/equalizer: 0 - sample_size: 0 - lr: 0.01 - batch_size: 128 - episodes: 100 - minimal_env: True/0/best-model')
storage = Storage()
mcts.draw_game_tree(rating, env, model, storage, 284, 5, [0])#495,185



