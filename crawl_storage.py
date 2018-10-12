
from src.KerasModel import KerasModel
from src.ModelRating import ModelRating
from src.Storage import Storage
from src.MCTS import MCTS
from gym_watten.envs.watten_env import WattenEnv
import pickle
from src.XorShfGenerator import XorShfGenerator
env = WattenEnv(True)
rating = ModelRating(env)
rng = XorShfGenerator()
mcts = MCTS(rng, episodes=10000, mcts_sims=100, exploration=0.1, step_exploration=0.9)
model = KerasModel(env, 128, equalizer=0, clip=0, batch_size=70214, policy_lr=100, policy_momentum=0.9, value_lr=0.005, value_momentum=0.9)
storage = Storage()
model.load('results/full_cards/clear_samples_after_epoch: True - episodes: 50 - mcts_sims: 100 - policy_lr: 0.5 - policy_momentum: 0.99 - step_exploration: 0.9 - value_lr: 0.005/5/best-model')

#mcts.mcts_generate(env, model, storage, rating)
#with open("storage.pk", 'wb') as handle:
#    pickle.dump(storage, handle)
with open("storage.pk", 'rb') as handle:
    storage = pickle.load(handle)

storage.export_csv('storage.csv', env)

model.memorize_storage(storage, False, 10000, 0)

mcts = MCTS(rng, episodes=1, mcts_sims=1, exploration=0.1, only_one_step=True)
p = mcts.draw_game_tree(rating, env, model, storage, 7, 5, [2])