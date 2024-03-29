# General imports
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

# API imports
from sc2.data import Difficulty, Race
from sc2.main import run_game
from sc2.player import Bot, Computer
from sc2 import maps

# Global variables to pick the right experiment and WandB project.
plotName = "ResourcePlot"
mapName = "TrainingMapResource"
episode_reward_list = []

#ArmyBot imports (depends on the chosen mapName above)
match mapName:
    case "TrainingMapMarine":
        from ArmyBotMarine import ArmyBot

    case "TrainingMapResource":
        from ArmyBotResource import ArmyBot

    case "TrainingMapBoth":
        from ArmyBotBoth import ArmyBot

# This is the thread that holds the queues and runs the game
class GameThread(Thread):
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

# This is the environment itself where Step and Reset are defined
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
                self.observation_space = MultiDiscrete([36, 36, 36, 1000, 2, 3])

            case "TrainingMapBoth":
                self.action_space = Discrete(5)
                # Values: [MarineHealth, ZergDist, WeaponCD, MarineNr, SCVNr, IdleSCVs, Minerals, CCAvailable, BarAvailable]
                self.observation_space = MultiDiscrete([46, 250, 2, 36, 36, 36, 1000, 2, 3])

            case _:
                print("You must choose a valid experiment")

    def step(self, action):
        # Send an action to the Bot
        self.gameThread.action_in.put(action)

        # Get the result
        out = self.gameThread.result_out.get()               
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
        self.gameThread = GameThread()
        self.gameThread.start()
        return observation, info


# This method is called at the end of each episode and is used to produce graphs in WandB
class WandBCallback(DefaultCallbacks):
    def on_episode_end(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        episode_reward = episode.total_reward

        global episode_reward_list
        episode_reward_list.append(episode_reward)
        listLength = len(episode_reward_list)
        print("End of episode", listLength)
        if listLength > 0 and listLength % 100 == 0:
            wandb.init(project=plotName)

            data = []

            for ep in range(len(episode_reward_list)):
                data.append([ep, episode_reward_list[ep]])
            print(data)    

            table = wandb.Table(data = data, columns = ["Episodes", "Rewards"])

            wandb.log({"episode_rewards" : wandb.plot.line(table, "Episodes", "Rewards", title="Custom Episode Rewards Line Plot")})
                
            wandb.finish()

# This method trains the agent using RLlib
def train_ppo():
    # Create a configuration for training
    config = PPOConfig()
    config = config.callbacks(callbacks_class=WandBCallback)
    config = config.training(entropy_coeff=0.01) # Entropy coeff determines how likeliy the algorithm is to explore new options
    config = config.resources(num_gpus=0)
    config = config.rollouts(num_rollout_workers=1)

    # Build the algorithm with the custom environment
    algo = config.build(env=QueueEnv)

    # Train the PPO agent
    iterations = 2000
    for i in range(iterations):  # Number of training iterations
        result = algo.train()

        if 0 == (i+1)%(iterations/8):
            checkpoint_dir = algo.save()
            print(f"Checkpoint saved in directory {checkpoint_dir}")


if __name__ == "__main__":    

    train_ppo()
    