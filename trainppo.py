from Sc2Env import *
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms import ppo
import time
import os
import ray

ray.init()
algo = ppo.PPO(env=Sc2Env)

#env = Sc2Env()

#onfig = (PPOConfig().environment(env=env).rollouts(num_rollout_workers=2))

#algo = config.build()

for i in range(5):
    print(algo.train())
