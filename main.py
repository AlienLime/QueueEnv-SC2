#General Python packages
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import asyncio
from threading import Thread
import time
from queue import Queue


#API imports
from sc2.bot_ai import BotAI  # parent class we inherit from
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

# 	#config = (PPOConfig().environment(env=SC2Env).rollouts(num_rollout_workers=1))

# 	#algo = config.build()

#     for i in range(1):
#         print(algo.train())

#     algo.stop()
#     print("main end") """

class AST(Thread):
    def __init__(self, loop) -> None:
        super().__init__()
        self.action_in = Queue()
        self.result_out = Queue()

        # self.bot_in_box = asyncio.Queue()
        self.loop = loop
        # self.bot = bot
    
    def run(self) -> None:
        print("setting loop")
        asyncio.set_event_loop(self.loop)
        print("starting game.")
        self.bot = ArmyBot(bot_in_box=self.bot_in_box)
        print("starting gamem.")
        run_game(  # run_game is a function that runs the game.
            maps.get("TestMap1"), # the map we are playing on
            [Bot(Race.Terran, self.bot), # runs our coded bot, Terran race, and we pass our bot object 
            Computer(Race.Zerg, Difficulty.Hard)], # runs a pre-made computer agent, zerg race, with a hard difficulty.
            realtime=False, # When set to True, the agent is limited in how long each step can take to process.
        )
        print("stopping game.")
        while True:
            break
            action = self.action_in.get()
            print("got an action", action)

            # print()
            self.bot.action = action
            # out_cr = self.bot.on_step(action)
            # print(f">>> out coroutine {out_cr=}")
            # out = asyncio.run(out_cr)
            if self.bot.output is None:
                print("Waiting for response")
            else:
                print("I got", self.bot.output) 
                output = self.bot.output
                self.bot.output = None

            # print(f">>> {out=}, putting in out queuue.")
            self.result_out.put(output)
            print(">>> all done laila tov.")
            time.sleep(0.2)


    def join() -> None:
        pass

class SC2Env(gym.Env):
        def __init__(self, config=None, render_mode=None): # None, "human", "rgb_array"
            super(SC2Env, self).__init__()
            # Define action and observation space
            # They must be gym.spaces objects
            # Example when using discrete actions:
            self.action_space = spaces.Discrete(2)
            self.observation_space = spaces.Box(low=0, high=255, shape=(224, 224, 3), dtype=np.uint8)


        def reset(self):
            print("RESETTING ENVIRONMENT!!!!!!!!!!!!!")
            map = np.zeros((224, 224, 3), dtype=np.uint8)
            observation = map
            
            self.pcom = AST(asyncio.get_event_loop())
            # self.pcom.start()

            # asyncio.set_event_loop(asyncio.new_event_loop())

            self.pcom.start()
            # assert False
            return observation, {}
        
        def step(self, action):
            # assert self.pcom.action_in.empty
            # assert action > 0 
            print("step, putting action.")
            self.pcom.action_in.put("A=action from step")
            print("step, waiting..")
            out = self.pcom.result_out.get()
            print("step, got", out)

            b =234

            
            # async def getResult():
            #     return await self.pcom.result_out.get()
            # print("puuuutt")
            # self.pcom.action_in.put(action)
            # # result = asyncio.run(getResult(action))
            # print("puuuutt")
            # result = self.pcom.result_out.get()
            # print("resuuuuult")
            # a = 34

            # result = getResult()

def ast_old():
    print("main JUBIIII")
    t = AST()
    t.start()
    async def addSome(action=1):
        t.value = action
        await t.action_in.put(action)
    asyncio.run(addSome(0))


    s = [addSome() for _ in range(10)]
    asyncio.run(addSome(0))


def run_sc2():
    env = SC2Env()
    s0, info = env.reset()
    # £print("first state is", s0)
    print( env.step(0) )


    

if __name__ == "__main__":

    run_sc2()

