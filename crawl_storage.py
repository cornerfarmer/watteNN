
from src.KerasModel import KerasModel
from src.ModelRating import ModelRating
from src.Storage import Storage
from src.LookUp import LookUp
from src.MCTS import MCTS
from gym_watten.envs.watten_env import WattenEnv
import pickle
from src.XorShfGenerator import XorShfGenerator
env = WattenEnv(True)
rating = ModelRating(env)
rng = XorShfGenerator()
mcts = MCTS(rng, episodes=10000, mcts_sims=10000, exploration=0.1, step_exploration=0.9)
#model = KerasModel(env, 128, equalizer=0, clip=0.15, batch_size=70214, policy_lr=100, policy_momentum=0.9, value_lr=0.005, value_momentum=0.9)
model = LookUp()
storage = Storage()
path = 'results/lookup/clear_samples_after_epoch: True - episodes: 50 - mcts_sims: 100 - policy_lr: 0.5 - policy_momentum: 0.99 - step_exploration: 0.9 - value_lr: 0.005/8'
#path += '/checkpoints/2800'
model.load(path + '/best-model')

mcts.mcts_generate(env, model, storage, rating)
#with open("storage.pk", 'wb') as handle:
#    pickle.dump(storage, handle)
#with open("storage.pk", 'rb') as handle:
#    storage = pickle.load(handle)

storage.export_csv('storage.csv', env)
storage.count_per_key('5.1,-4,7,13,')
exit(0)
model.memorize_storage(storage, False, 1, 0)


mcts = MCTS(rng, episodes=1, mcts_sims=1, exploration=0.1, only_one_step=True)
p = mcts.draw_game_tree(rating, env, model, storage, 7, 5, [])