#General Python packages
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Discrete
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

#Custom imports
from ArmyBot import ArmyBot

class AST(Thread):
    def __init__(self) -> None:
        super().__init__()
        self.action_in = Queue()
        self.result_out = Queue()
    
    def run(self) -> None:
        self.bot = ArmyBot(action_in=self.action_in, result_out=self.result_out)
        print("starting game.")
        result = run_game(  # run_game is a function that runs the game.
            maps.get("TestMap3"), # the map we are playing on
            [Bot(Race.Terran, self.bot), # runs our coded bot, Terran race, and we pass our bot object 
            Computer(Race.Zerg, Difficulty.Hard)], # runs a pre-made computer agent, zerg race, with a hard difficulty.
            realtime=False, # When set to True, the agent is limited in how long each step can take to process.
        )



class QueueEnv(gym.Env):
    def __init__(self, config=None, render_mode=None): # None, "human", "rgb_array"
        super(QueueEnv, self).__init__()
        # Define action and observation space
        self.action_space = Discrete(2)

        # Values: [MarineHealth, ZergDist, WeaponCD, MarineNr, SCVNr, Minerals]
        self.observation_space = MultiDiscrete([46,250,2,20,20,1000])
    
    def step(self, action):
        # Send an action to the Bot
        self.pcom.action_in.put(action)

        # Get the result
        out = self.pcom.result_out.get()               
        observation = out["observation"]
        reward = out["reward"]
        done = out["done"]
        truncated = out["truncated"]
        info = out["info"]

        return observation, reward, done, truncated, info
    
    def reset(self, *, seed=None, options=None):
        print("RESETTING ENVIRONMENT!!!!!!!!!!!!!")
        time.sleep(5)

        # Values: [MarineHealth, ZergDist, WeaponCD, MarineNr, SCVNr, Minerals]
        observation = np.array([45, 210, 0, 1, 1, 50])
        info = {}
        self.pcom = AST()
        self.pcom.start()
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

    # Create a configuration for training
    config = PPOConfig()
    config = config.training(entropy_coeff = 0.01) # Entropy coeff determines how likeliy the algorithm is to explore new options
    config = config.resources(num_gpus=0)
    config = config.rollouts(num_rollout_workers=1)

    # Build the algorithm with the custom environment
    algo = config.build(env=QueueEnv)

    # Train the PPO agent
    iterations = 10
    for i in range(iterations):  # Number of training iterations
        result = algo.train()
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