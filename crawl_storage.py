
from src.KerasModel import KerasModel
from src.ModelRating import ModelRating
from src.Storage import Storage
from src.MCTS import MCTS
from gym_watten.envs.watten_env import WattenEnv

env = WattenEnv(True)
rating = ModelRating(env)
mcts = MCTS(episodes=1000, mcts_sims=40, exploration=0.1)
model = KerasModel(env, 128)
storage = Storage()
model.load('results/exploration/batch_size: 128 - lr: 0.1 - minimal_env: True - sample_size: 0 - storage_size: 80000/0/best-model')

mcts.mcts_generate(env, model, storage)
storage.export_csv('storage.csv', env)

model.memorize_storage(storage, False, 1000, 0)

p = mcts.draw_game_tree(rating, env, model, 335, 5, [1])