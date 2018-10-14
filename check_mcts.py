
from src.KerasModel import KerasModel
from src.ModelRating import ModelRating
from src.MCTS import MCTS
from src.Storage import Storage
from gym_watten.envs.watten_env import WattenEnv

from src.XorShfGenerator import XorShfGenerator

env = WattenEnv(True)
rating = ModelRating(env)
rng = XorShfGenerator()
mcts = MCTS(rng, episodes=1, mcts_sims=100, exploration=0.1, only_one_step=True, step_exploration=0)
model = KerasModel(env, 128, clip=0.15)
path = 'results/very_minimal/clear_samples_after_epoch: True - episodes: 50 - mcts_sims: 100 - policy_lr: 0.5 - policy_momentum: 0.99 - step_exploration: 0.9 - value_lr: 0.005/2'
path += '/checkpoints/2800'
model.load(path + '/best-model')

storage = Storage()
mcts.draw_game_tree(rating, env, model, storage, 49, 5, [1,1])#495,185



