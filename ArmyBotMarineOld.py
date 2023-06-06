from sc2.bot_ai import BotAI  # parent class we inherit from
#from sc2.data import Difficulty, Race  # difficulty for bots, race for the 1 of 3 races
#from sc2.main import run_game  # function that facilitates actually running the agents in games
#from sc2.player import Bot, Computer  #wrapper for whether or not the agent is one of your bots, or a "computer" player
#from sc2 import maps  # maps method for loading maps to play in.
from sc2.ids.unit_typeid import UnitTypeId

import sc2.main
import numpy as np
import random
import cv2
import math
import time
import sys
# import asyncio



class ArmyBot(BotAI): # inhereits from BotAI (part of BurnySC2)
    action = None
    output = {"observation" : map, "reward" : 0, "action" : None, "done" : False}
    def __init__(self, *args,bot_in_box=None, action_in=None, result_out=None, **kwargs, ):
        super().__init__(*args, **kwargs)
        self.action_in = action_in
        self.result_out = result_out

    async def on_end(self,game_result):
        print ("Game over!")
        reward = 0
        info = {"hp" : None}
        map = np.zeros((42, 42, 3), dtype=np.uint8)
        cv2.destroyAllWindows()
        #cv2.waitKey(1)
        
        if str(game_result) == "Result.Victory":
            for marine in self.units(UnitTypeId.MARINE):
                if marine.health_percentage < 1:
                    reward += marine.health
                    info["hp"] = marine.health
        else:
            reward = -10


        self.result_out.put({"observation" : map, "reward" : reward, "action" : None, "done" : True, "truncated" : False, "info" : info})
        
    
    async def on_step(self, iteration): # on_step is a method that is called every step of the game.
        self.action = self.action_in.get()
        '''
        0: Force Move
        1: Attack Move
        '''
        if iteration % 10 == 0:
            print("armybot at...", iteration)
        # if self.bot_in_box is not None:
        #     action = self.bot_in_box.get()
        #     print("action,", action)
        #     action = asyncio.run(action)
        #print("Got action from outside", self.action, "I will now execute that action.")
        # print("<updating...")
        # This gets an action and returns a state. You probably need to put logic here such as waiting a certain amount of in-game time before retuning etc. (you
        # don't want the states to be 'too close' if that makes sense)

        if self.action is None:
            # print("no action returning.")
            return None
        time.sleep(0.05)
        # 0: Force Move
        #print("Action is", self.action)
        if self.action == 0:
            try:
                for marine in self.units(UnitTypeId.MARINE):
                    for sd in self.structures(UnitTypeId.SUPPLYDEPOT):
                        marine.move(sd)
            except Exception as e:
                print(e)

        #1: Attack Move
        elif self.action == 1:
            try:
                for marine in self.units(UnitTypeId.MARINE):
                    marine.attack(random.choice(self.enemy_units))
            except Exception as e:
                print(e)

        #print("returning a result from army bot..")
        
        map = np.zeros((42, 42, 3), dtype=np.uint8)

        # draw the minerals:
        for mineral in self.mineral_field:
            pos = mineral.position
            c = [175, 255, 255]
            fraction = mineral.mineral_contents / 1800
            if mineral.is_visible:
                #print(mineral.mineral_contents)
                map[math.ceil(pos.y)][math.ceil(pos.x)] = [int(fraction*i) for i in c]
            else:
                map[math.ceil(pos.y)][math.ceil(pos.x)] = [20,75,50]  

        # draw the enemy units:
        for enemy_unit in self.enemy_units:
            pos = enemy_unit.position
            c = [100, 0, 255]
            # get unit health fraction:
            fraction = enemy_unit.health / enemy_unit.health_max if enemy_unit.health_max > 0 else 0.0001
            map[math.ceil(pos.y)][math.ceil(pos.x)] = [int(fraction*i) for i in c]

        # draw our structures:
        for our_structure in self.structures:
            # if it's a commandcenter:
            if our_structure.type_id == UnitTypeId.COMMANDCENTER:
                pos = our_structure.position
                c = [255, 255, 175]
                # get structure health fraction:
                fraction = our_structure.health / our_structure.health_max if our_structure.health_max > 0 else 0.0001
                map[math.ceil(pos.y)][math.ceil(pos.x)] = c
            # if it's a barracks
            elif our_structure.type_id == UnitTypeId.BARRACKS:
                pos = our_structure.position
                c = [125, 255, 175]
                map[math.ceil(pos.y)][math.ceil(pos.x)] = c

            else:
                pos = our_structure.position
                c = [0, 255, 175]
                map[math.ceil(pos.y)][math.ceil(pos.x)] = c

        # draw our units:
        for our_unit in self.units:
            # if it is a marine:
            if our_unit.type_id == UnitTypeId.MARINE:
                pos = our_unit.position
                c = [255, 75 , 75]
                # get health:
                fraction = our_unit.health / our_unit.health_max if our_unit.health_max > 0 else 0.0001
                map[math.ceil(pos.y)][math.ceil(pos.x)] = [int(fraction*i) for i in c]


            else:
                pos = our_unit.position
                c = [175, 255, 0]
                map[math.ceil(pos.y)][math.ceil(pos.x)] = c

        # show map with opencv, resized to be larger:
        # horizontal flip:

        if iteration != 0:
            cv2.imshow('map',cv2.flip(cv2.resize(map, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST), 0))
            cv2.waitKey(1)

        reward = -1

        try:
            # iterate through our marines:
            for marine in self.units(UnitTypeId.MARINE):
                # if marine is attacking and is in range of enemy unit:
                if marine.is_attacking:
                    reward += 1
                    if self.enemy_units.closer_than(5, marine) and marine.weapon_cooldown == 0:
                        reward += 1
                else:
                    if marine.weapon_cooldown > 0:
                        reward += 2

        except Exception as e:
            print("reward",e)
            reward = 0

        truncated = False
        if iteration == 500:

            truncated = True

        self.result_out.put({"observation" : map, "reward" : reward, "action" : None, "done" : False, "truncated" : truncated, "info" : {}})
        


#time.sleep(3)
#sys.exit()