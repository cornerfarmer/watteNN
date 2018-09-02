
from src.KerasModel import KerasModel
from src.ModelRating import ModelRating
from src.Storage import Storage
from src.MCTS import MCTS
from gym_watten.envs.watten_env import WattenEnv
import pickle
env = WattenEnv(True)
rating = ModelRating(env)
mcts = MCTS(episodes=10000, mcts_sims=40, exploration=0.1, step_exploration=0.1)
model = KerasModel(env, 128, equalizer=0, clip=0.15, batch_size=128, lr=0.01)
storage = Storage()
model.load('results/step_exploration/equalizer: 0 - sample_size: 0 - lr: 0.01 - batch_size: 128 - episodes: 100 - minimal_env: True/1/best-model')

mcts.mcts_generate(env, model, storage)
#with open("storage.pk", 'wb') as handle:
#    pickle.dump(storage, handle)
#with open("storage.pk", 'rb') as handle:
#    storage = pickle.load(handle)

storage.export_csv('storage.csv', env)
exit(0)
model.memorize_storage(storage, False, 10, 0)

mcts = MCTS(episodes=1, mcts_sims=40, exploration=0.1, only_one_step=True)
p = mcts.draw_game_tree(rating, env, model, storage, 110, 5, [1, 2])