
from src.KerasModel import KerasModel
from src.Game import Game
from src.ModelRating import ModelRating
from src.LookUp import LookUp
from gym_watten.envs.watten_env import WattenEnv


env = WattenEnv(True)
rating = ModelRating(env)
model = LookUp()#KerasModel(env, 128, clip=0.15)
path = 'results/lookup/clear_samples_after_epoch: True - episodes: 50 - mcts_sims: 100 - policy_lr: 0.5 - policy_momentum: 0.99 - step_exploration: 0.9 - value_lr: 0.005/17'
#path += '/checkpoints/2800'
model.load(path + '/best-model')

game = Game(env)
tree_ind = 170
table, avg_diff_on, avg_diff_off, max_diff, v_loss_on, v_loss_off, v_based_avg_diff_on, v_based_avg_diff_off = game.draw_game_tree(model, rating, False, tree_ind, ",15.1-5,6,7,", tree_path=path + "/tree-" + str(tree_ind) + ".svg")#335,249,284

#print(table[',13-4,12,15,'])
print(avg_diff_on, avg_diff_off, max_diff)
print("v_loss on/off: " + str(v_loss_on) + ", " + str(v_loss_off))
