#General Python packages
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Discrete
import numpy as np
from threading import Thread
import time
from queue import Queue
import wandb

# Ray imports
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks

#API imports
from sc2.data import Difficulty, Race  # difficulty for bots, race for the 1 of 3 races
from sc2.main import run_game  # function that facilitates actually running the agents in games
from sc2.player import Bot, Computer  #wrapper for whether or not the agent is one of your bots, or a "computer" player
from sc2 import maps  # maps method for loading maps to play in.

# Global variables to pick the right experiment and WandB project.
projectName = "ArmyBot2"
mapName = "TrainingMapResource"

#Custom imports
match mapName:
    case "TrainingMapMarine":
        from ArmyBotMarine import ArmyBot

    case "TrainingMapResource":
        from ArmyBotResource import ArmyBot

    case "TrainingMapBoth":
        from ArmyBotBoth import ArmyBot

class AST(Thread):
    def __init__(self) -> None:
        super().__init__()
        self.action_in = Queue()
        self.result_out = Queue()
    
    def run(self) -> None:
        self.bot = ArmyBot(action_in=self.action_in, result_out=self.result_out)
        print("starting game.")
        result = run_game(  # run_game is a function that runs the game.
            maps.get(mapName), # the map we are playing on
            [Bot(Race.Terran, self.bot), # runs our coded bot, Terran race, and we pass our bot object 
            Computer(Race.Zerg, Difficulty.Hard)], # runs a pre-made computer agent, zerg race, with a hard difficulty.
            realtime=False, # When set to True, the agent is limited in how long each step can take to process.
        )


class QueueEnv(gym.Env):
    def __init__(self, config=None, render_mode=None): # None, "human", "rgb_array"
        super(QueueEnv, self).__init__()

        # Define action and observation space
        match mapName:
            case "TrainingMapMarine":
                self.action_space = Discrete(2)
                # Values: [MarineHealth, ZergDist, WeaponCD]
                self.observation_space = MultiDiscrete([46, 250, 2])

            case "TrainingMapResource":
                self.action_space = Discrete(3)
                # Values: [MarineNr, SCVNr, IdleSCVs, Minerals, CCAvailable, BarAvailable]
                self.observation_space = MultiDiscrete([20, 20, 20,1000, 2, 3])

            case "TrainingMapBoth":
                self.action_space = Discrete(5)
                # Values: [MarineHealth, ZergDist, WeaponCD, MarineNr, SCVNr, IdleSCVs, Minerals, CCAvailable, BarAvailable]
                self.observation_space = MultiDiscrete([46, 250, 2, 25, 20, 20, 1000, 2, 3])

            case _:
                print("You must choose a valid experiment")

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
        print("!RESETTING ENVIRONMENT!")
        time.sleep(5)
        
        match mapName:
            case "TrainingMapMarine":
                # Values: [MarineHealth, ZergDist, WeaponCD]
                observation = np.array([45, 210, 0])

            case "TrainingMapResource":
                # Values: [MarineNr, SCVNr, IdleSCVs, Minerals, CCAvailable, BarAvailable]
                observation = np.array([1, 1, 1, 50, 1, 2])

            case "TrainingMapBoth":
                # Values: [MarineHealth, ZergDist, WeaponCD, MarineNr, SCVNr, IdleSCVs, Minerals, CCAvailable, BarAvailable]
                observation = np.array([45, 210, 0, 1, 1, 1, 50, 1, 2])

            case _:
                print("You must choose a valid experiment")

        info = {}
        self.pcom = AST()
        self.pcom.start()
        return observation, info

class WandBCallback(DefaultCallbacks):
    def on_episode_end(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        episode_reward = episode.total_reward
        wandb.run.log({"Episode Reward": episode_reward})

def train_ppo():
    # Initialize WandB
    wandb.init(project = projectName)

    # Create a configuration for training
    config = PPOConfig()
    config = config.callbacks(callbacks_class=WandBCallback)
    config = config.training(entropy_coeff=0.01) # Entropy coeff determines how likeliy the algorithm is to explore new options
    config = config.resources(num_gpus=0)
    config = config.rollouts(num_rollout_workers=1)

    # Build the algorithm with the custom environment
    algo = config.build(env=QueueEnv)

    # Train the PPO agent
    iterations = 2
    for i in range(iterations):  # Number of training iterations
        result = algo.train()

        if i == iterations - 1:
            checkpoint_dir = algo.save()
            print(f"Checkpoint saved in directory {checkpoint_dir}")
            
    wandb.finish()

if __name__ == "__main__":    

    train_ppo()

