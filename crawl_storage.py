
from src.KerasModel import KerasModel
from src.ModelRating import ModelRating
from src.Storage import Storage
from src.MCTS import MCTS
from gym_watten.envs.watten_env import WattenEnv
import pickle
env = WattenEnv(True)
rating = ModelRating(env)
mcts = MCTS(episodes=10000, mcts_sims=40, exploration=0.1, step_exploration=0.1)
model = KerasModel(env, 128, equalizer=0, clip=0.15, batch_size=128, policy_lr=0.01, policy_momentum=0, value_lr=0.1, value_momentum=0.9)
storage = Storage()
model.load('results/true_loss/sample_size: 0 - value_momentum: 0.3 - batch_size: 128 - equalizer: 0 - policy_lr: 0.01 - episodes: 100 - value_lr: 0.001 - minimal_env: True/2/model')

mcts.mcts_generate(env, model, storage, rating)
#with open("storage.pk", 'wb') as handle:
#    pickle.dump(storage, handle)
#with open("storage.pk", 'rb') as handle:
#    storage = pickle.load(handle)

storage.export_csv('storage.csv', env)
exit(0)
#model = KerasModel(env, 128, equalizer=0, clip=0.15, batch_size=128, policy_lr=0.01, policy_momentum=0, value_lr=0.01, value_momentum=0.9)
model.memorize_storage(storage, False, 1000, 0)

mcts = MCTS(episodes=1, mcts_sims=80, exploration=0.1, only_one_step=True)
p = mcts.draw_game_tree(rating, env, model, storage, 230, 5, [2])