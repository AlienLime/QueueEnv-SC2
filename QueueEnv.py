#General Python packages
import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import numpy as np
import os
import asyncio
from threading import Thread
import time
from queue import Queue
from wandb.keras import WandbCallback
from matplotlib import pyplot as plt
import wandb
import sys

# Ray imports
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.misc import normc_initializer
#from ray.rllib.utils import try_import_tf
#tf = try_import_tf()


#API imports
from sc2.data import Difficulty, Race  # difficulty for bots, race for the 1 of 3 races
from sc2.main import run_game  # function that facilitates actually running the agents in games
from sc2.player import Bot, Computer  #wrapper for whether or not the agent is one of your bots, or a "computer" player
from sc2 import maps  # maps method for loading maps to play in.
from sc2.ids.unit_typeid import UnitTypeId


#Reinforcement learning packages
#import ray
#from ray.rllib.algorithms.ppo import PPOConfig
#from ray.rllib.algorithms import ppo

#Custom imports
from ArmyBotTerran import ArmyBot

# async def main(action_in, result_out):
#     for action in [1,0]:
#         await action_in.put(action)
#     print("put all actions")
#     # Run the tasks
#     await asyncio.gather(
#         asyncio.create_task(ArmyBot.on_step("One", action_in, result_out)),
#         asyncio.create_task(ArmyBot.on_step("Two", action_in, result_out))
#         )
#     print("gathered all")

#     """  ray.init()
#     print("main start")

#     algo = ppo.PPO(env=SC2Env)

#     env = SC2Env()

#  #config = (PPOConfig().environment(env=SC2Env).rollouts(num_rollout_workers=1))

#  #algo = config.build()

#     for i in range(1):
#         print(algo.train())

#     algo.stop()
#     print("main end") """

class AST(Thread):
    def __init__(self) -> None:
        super().__init__()
        self.action_in = Queue()
        self.result_out = Queue()

        # self.bot_in_box = asyncio.Queue()
        # self.loop = loop
        # self.bot = bot
    
    def run(self) -> None:
        # print("setting loop")
        #loop = asyncio.new_event_loop()
        #asyncio.set_event_loop(loop)
        # print("starting game.")
        self.bot = ArmyBot(action_in=self.action_in, result_out=self.result_out)
        print("starting game.")
        result = run_game(  # run_game is a function that runs the game.
            maps.get("TestMap3"), # the map we are playing on
            [Bot(Race.Terran, self.bot), # runs our coded bot, Terran race, and we pass our bot object 
            Computer(Race.Zerg, Difficulty.Hard)], # runs a pre-made computer agent, zerg race, with a hard difficulty.
            realtime=True, # When set to True, the agent is limited in how long each step can take to process.
        )



class QueueEnv(gym.Env):
    def __init__(self, config=None, render_mode=None): # None, "human", "rgb_array"
        super(QueueEnv, self).__init__()
        # Define action and observation space
        self.action_space = Discrete(2)
        self.observation_space = Box(low=0, high=255, shape=(42, 42, 3), dtype=np.uint8)
    
    def step(self, action):
        # Send an action to the Bot
        self.pcom.action_in.put(action)

        # Get the result
        out = self.pcom.result_out.get()               
        observation = out["observation"]
        reward = out["reward"]
        done = out["done"]
        truncated = out["truncated"]
        info = {}

        return observation, reward, done, truncated, info
    
    def reset(self, *, seed=None, options=None):
        print("RESETTING ENVIRONMENT!!!!!!!!!!!!!")
        time.sleep(5)
        map = np.zeros((42, 42, 3), dtype=np.uint8)
        observation = map
        info = {}
        self.pcom = AST()
        self.pcom.start()
        # assert False
        return observation, info
    

def run_sc2():
    env = QueueEnv()
    
    for _ in range(3):
        print("iteration", _)
        s, info = env.reset()
        while True:
            a = env.action_space.sample()
            if np.random.rand() < 0.6: 
                a = 1
            s, r, done, truncated, info = env.step(a)
            if done or truncated:
                print("ran one episode")
                break

def train_ppo():
    # Initialize WandB
    wandb.init(project="ArmyBot1")

    # Make a list
    rewards = []

    # Create an algo and configure it with your custom environment
    algo = (
        PPOConfig()
        .rollouts(num_rollout_workers=1)
        .resources(num_gpus=0)
        .environment(env=QueueEnv)
        .build()
    )

    # Train the PPO agent
    iterations = 2
    for i in range(iterations):  # Number of training iterations
        result = algo.train()
        #print(result)
        episode_reward = result["hist_stats"]["episode_reward"]

        for rwd in episode_reward:
            rewards.append(rwd)
        
        

        if i == iterations - 1:
            checkpoint_dir = algo.save()
            print(f"Checkpoint saved in directory {checkpoint_dir}")
            data = []

            for ep in range(len(rewards)):
                data.append([ep, rewards[ep]])

            table = wandb.Table(data = data, columns = ["Episodes", "Rewards"])

            wandb.log({"episode_rewards" : wandb.plot.line(table, "Episodes", "Rewards", title="Custom Episode Rewards Line Plot")})

    wandb.finish()


if __name__ == "__main__":    
    #test
    #run_sc2()

    #train
    train_ppo()