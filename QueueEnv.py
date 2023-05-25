#General Python packages
import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import numpy as np
import os
import asyncio
from threading import Thread
import time
from queue import Queue
from wandb.integration.sb3 import WandbCallback
from matplotlib import pyplot as plt
import wandb

# Ray imports
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.utils import try_import_tf
tf = try_import_tf()


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
from ArmyBot import ArmyBot

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
            maps.get("BerlingradAIE"), # the map we are playing on
            [Bot(Race.Protoss, self.bot), # runs our coded bot, Terran race, and we pass our bot object 
            Computer(Race.Zerg, Difficulty.Hard)], # runs a pre-made computer agent, zerg race, with a hard difficulty.
            realtime=False, # When set to True, the agent is limited in how long each step can take to process.
        )
        # print("stopping game.")
        # while True:
        #     break
        #     action = self.action_in.get()
        #     print("got an action", action)
        #
        #     # print()
        #     self.bot.action = action
        #     # out_cr = self.bot.on_step(action)
        #     # print(f">>> out coroutine {out_cr=}")
        #     # out = asyncio.run(out_cr)
        #     if self.bot.output is None:
        #         print("Waiting for response")
        #     else:
        #         print("I got", self.bot.output)
        #         output = self.bot.output
        #         self.bot.output = None
        #
        #     # print(f">>> {out=}, putting in out queuue.")
        #     self.result_out.put(output)
        #     print(">>> all done laila tov.")
        #     time.sleep(0.2)


    # def join() -> None:
    #     pass

class QueueEnv(gym.Env):
        def __init__(self, config=None, render_mode=None): # None, "human", "rgb_array"
            super(QueueEnv, self).__init__()
            # Define action and observation space
            self.action_space = Discrete(6)
            self.observation_space = Box(low=0, high=255, shape=(152, 168, 3), dtype=np.uint8)

            # Helper variables for time data
            self.iteration = -1
            self.starttime = 0
            self.timeData = []
            self.stepList = []
        
        def step(self, action):
            #increment iteration
            self.iteration += 1

            #start a timer
            if self.iteration % 100 == 1:
                self.starttime = time.time()

            # print("SC2.step()> putting action.")
            self.pcom.action_in.put(action)
            # print("step, waiting..")
            out = self.pcom.result_out.get()

            #Get the time taken
            if self.iteration % 100 == 0 and self.iteration > 0:
                steptime = round(time.time() - self.starttime, 2)
                if not(self.iteration == 2100 or self.iteration == 4100 or self.iteration == 6200 or self.iteration == 8200):
                    self.timeData.append(steptime)
                    self.stepList.append(self.iteration)
                print("These 100 steps took", steptime, "seconds")
                if self.iteration == 6500:
                    plt.plot(self.stepList, self.timeData)
                    plt.ylim(0, max(self.timeData) + 1)
                    plt.show()
                    

            observation = out["observation"]
            reward = out["reward"]
            done = out["done"]
            truncated = False
            info = {}
            return observation, reward, done, truncated, info
        
        def reset(self, *, seed=None, options=None):
            print("RESETTING ENVIRONMENT!!!!!!!!!!!!!")
            map = np.zeros((152, 168, 3), dtype=np.uint8)
            observation = map
            info = {}
            self.pcom = AST()
            # self.pcom.start()
            # asyncio.set_event_loop(asyncio.new_event_loop())
            self.pcom.start()
            # assert False
            return observation, info

def run_sc2():
    env = QueueEnv()
    s0, info = env.reset()
    # env.pcom.action_in.put("something")
    # £print("first state is", s0)
    go = 0
    while go < 100:
        print("Step", go)
        sp, reward, done, info = env.step(go%2)
        print("Next state was:", go)
        go+=1
    print("done ya grease")

# Custom model class
class CustomModel(TFModelV2):
    def __init__(self, obs_space, action_space, model_config, name):
        super(CustomModel, self).__init__(obs_space, action_space, model_config, name)
        
        # Define the convolutional layers
        self.conv1 = tf.keras.layers.Conv2D(16, [3, 3], strides=(1, 1), activation=tf.nn.relu, padding="same",
                                            kernel_initializer=normc_initializer(1.0))
        self.conv2 = tf.keras.layers.Conv2D(32, [3, 3], strides=(1, 1), activation=tf.nn.relu, padding="same",
                                            kernel_initializer=normc_initializer(1.0))
        self.conv3 = tf.keras.layers.Conv2D(64, [3, 3], strides=(1, 1), activation=tf.nn.relu, padding="same",
                                            kernel_initializer=normc_initializer(1.0))
        
        # Define the fully connected layer
        self.fc = tf.keras.layers.Dense(action_space.n, activation=None,
                                        kernel_initializer=normc_initializer(0.01))

    def forward(self, input_dict, state, seq_lens):
        # Get the input observations
        obs = input_dict["obs"]
        
        # Apply the convolutional layers
        conv_out = self.conv1(obs)
        conv_out = self.conv2(conv_out)
        conv_out = self.conv3(conv_out)
        
        # Flatten the convolutional output
        flattened = tf.keras.layers.Flatten()(conv_out)
        
        # Apply the fully connected layer
        logits = self.fc(flattened)
        
        return logits, state

def train_ppo():
    # Register the custom model class
    ModelCatalog.register_custom_model("custom_model", CustomModel)

    # Create an instance of your custom environment
    env = QueueEnv()

    # Create a PPOTrainer and configure it with your custom environment
    config = {
        "env": QueueEnv,
        "framework": "tf",  # Use "tf" for TensorFlow
        "num_workers": 1,
        "model": {
            "custom_model": "custom_model",
            "custom_model_config": {
                "conv_filters": [[16, [3, 3], 1], [32, [3, 3], 1], [64, [3, 3], 1]],
            },
        },
    }
    trainer = PPOTrainer(config)

    # Train the PPO agent
    result = trainer.train()
    print(result)

    # Get the best trained agent
    best_agent = trainer.get_policy().model

    # Use the trained agent to interact with the environment
    obs = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = best_agent.action(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward

    print("Total reward:", total_reward)



if __name__ == "__main__":

    train_ppo()