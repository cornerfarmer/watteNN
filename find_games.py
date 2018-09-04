from src.ModelRating import ModelRating
from gym_watten.envs.watten_env import WattenEnv

env = WattenEnv(True)
rating = ModelRating(env)
print(rating.find("EO,EK,GO,", "EU,EA,GU,"))