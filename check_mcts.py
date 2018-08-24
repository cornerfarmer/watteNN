
from src.KerasModel import KerasModel
from src.ModelRating import ModelRating
from src.MCTS import MCTS
from src.Storage import Storage
from gym_watten.envs.watten_env import WattenEnv

env = WattenEnv(True)
rating = ModelRating(env)
mcts = MCTS(episodes=1, mcts_sims=40, exploration=0.4, only_one_step=True)
model = KerasModel(env, 128, clip=2)
model.load('results/momentumFix/batch_size: 128 - episodes: 100 - equalizer: 0.001 - lr: 0.1 - minimal_env: True - sample_size: 0/0/best-model')
storage = Storage()
mcts.draw_game_tree(rating, env, model, storage, 335, 5, [1])#495



