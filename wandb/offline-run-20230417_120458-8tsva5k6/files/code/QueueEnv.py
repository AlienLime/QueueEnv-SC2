#General Python packages
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import asyncio
from threading import Thread
import time
from queue import Queue
from wandb.integration.sb3 import WandbCallback
import wandb
from stable_baselines3 import PPO


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
        run_game(  # run_game is a function that runs the game.
            maps.get("WaterfallAIE"), # the map we are playing on
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
            # They must be gym.spaces objects
            # Example when using discrete actions:
            self.action_space = spaces.Discrete(6)
            self.observation_space = spaces.Box(low=0, high=255, shape=(144, 160, 3), dtype=np.uint8)
        
        def step(self, action):
            # assert self.pcom.action_in.empty
            # assert action > 0 
            print("SC2.step()> putting action.")
            self.pcom.action_in.put(action)
            # print("step, waiting..")
            out = self.pcom.result_out.get()
            print("SC2.step().step(), got", out)
            observation = out["observation"]
            reward = out["reward"]
            done = out["done"]
            info = {}
            return observation, reward, done, info
        
        def reset(self):
            print("RESETTING ENVIRONMENT!!!!!!!!!!!!!")
            map = np.zeros((144, 160, 3), dtype=np.uint8)
            observation = map
            self.pcom = AST()
            # self.pcom.start()
            # asyncio.set_event_loop(asyncio.new_event_loop())
            self.pcom.start()
            # assert False
            return observation, {}

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

def train_ppo():
    model_name = f"{int(time.time())}"

    models_dir = f"models/{model_name}/"
    logdir = f"logs/{model_name}/"


    conf_dict = {"Model": "v19",
                "Machine": "Main",
                "policy":"MlpPolicy",
                "model_save_name": model_name}


    run = wandb.init(
        project=f'SC2RLv6',
        entity="KrisEmil",
        config=conf_dict,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        save_code=True,  # optional
    )


    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    env = QueueEnv()

    model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)

    TIMESTEPS = 10000
    iters = 0
    while True:
        print("On iteration: ", iters)
        iters += 1
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
        model.save(f"{models_dir}/{TIMESTEPS*iters}")

if __name__ == "__main__":

    train_ppo()