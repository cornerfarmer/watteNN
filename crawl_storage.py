
from src.KerasModel import KerasModel
from src.ModelRating import ModelRating
from src.Storage import Storage
from src.MCTS import MCTS
from gym_watten.envs.watten_env import WattenEnv

env = WattenEnv(True)
rating = ModelRating(env)
mcts = MCTS(episodes=1000, mcts_sims=40)
model = KerasModel(env, 128)
storage = Storage()
model.load('results/maskedSoftmax/minimal_env: True - sample_size: 3500/0/best-model')

mcts.mcts_generate(env, model, storage)
storage.export_csv('storage.csv', env)
