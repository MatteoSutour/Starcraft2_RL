from stable_baselines3 import PPO
import os
from sc2env import Sc2Env
import time

os.environ["SC2PATH"] = "~/StarCraftII/"
os.environ["SC2_WSL_DETECT"] = "0"
os.environ['DISPLAY'] = ':0'

model_name = f"{int(time.time())}"
# model_name = "1678978294"

models_dir = f"models/{model_name}/"
# logdir = f"logs/{model_name}/"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# if not os.path.exists(logdir):
#     os.makedirs(logdir)

env = Sc2Env()

model = PPO('MlpPolicy', env, verbose=1)
# model = model.load("./models/1678978294/780.zip", env)

TIMESTEPS = 10
iters = 0
while True:
    print("On iteration: ", iters)
    iters += 1
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
    model.save(f"{models_dir}/{TIMESTEPS * iters}")
