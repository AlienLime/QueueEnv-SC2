from gym import spaces
import numpy as np
from threading import Thread
from queue import Queue

from IncrediBot import IncrediBot

#API imports
from sc2.bot_ai import BotAI  # parent class we inherit from
from sc2.data import Difficulty, Race  # difficulty for bots, race for the 1 of 3 races
from sc2.main import run_game  # function that facilitates actually running the agents in games
from sc2.player import Bot, Computer  #wrapper for whether or not the agent is one of your bots, or a "computer" player
from sc2 import maps  # maps method for loading maps to play in.
from sc2.ids.unit_typeid import UnitTypeId

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
        self.bot = IncrediBot(action_in=self.action_in, result_out=self.result_out)
        print("starting gamem.")
        result = run_game(  # run_game is a function that runs the game.
			maps.get("WaterfallAIE"), # the map we are playing on
			[Bot(Race.Protoss, self.bot), # runs our coded bot, Terran race, and we pass our bot object 
            Computer(Race.Zerg, Difficulty.Hard)], # runs a pre-made computer agent, zerg race, with a hard difficulty.
            realtime=False, # When set to True, the agent is limited in how long each step can take to process.
        )
        if str(result) == "Result.Victory":
            rwd = 500
        else:
            rwd = -500
        
        map = np.zeros((144, 160, 3), dtype=np.uint8)
        self.result_out.put({"observation": map, "reward": rwd, "action": None, "done": True})