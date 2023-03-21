from Sc2EnvOld import *
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms import ppo
import time
import os
import ray

ray.init()
#algo = ppo.PPO(env=Sc2Env)

#env = Sc2Env()

config = (PPOConfig().environment(env=Sc2Env).rollouts(num_rollout_workers=1))

algo = config.build()

for i in range(1):
    print(algo.train())

algo.stop()