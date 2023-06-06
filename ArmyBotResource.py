from sc2.bot_ai import BotAI  # parent class we inherit from
from sc2.ids.unit_typeid import UnitTypeId

import numpy as np
import random
import time




class ArmyBot(BotAI): # inhereits from BotAI (part of BurnySC2)
    action = None
    output = {"observation" : map, "reward" : 0, "action" : None, "done" : False}
    def __init__(self, *args,bot_in_box=None, action_in=None, result_out=None, **kwargs, ):
        super().__init__(*args, **kwargs)
        self.tickRate = 0.05
        self.action_in = action_in
        self.result_out = result_out

    async def on_end(self,game_result):
        print ("Game over!")
        
        # Values: [MarineNr, SCVNr, Minerals, CCAvailable, BarAvailable]
        obs = np.zeros(5, dtype=np.uint16)
        obs[0] = self.army_count
        obs[1] = self.supply_workers
        obs[2] = self.minerals
        for cc in self.structures(UnitTypeId.COMMANDCENTER).idle:
            obs[3] += 1
        for bar in self.structures(UnitTypeId.BARRACKS).idle:
            obs[4] += 1
        
        if str(game_result) == "Result.Victory":
            reward = 100
                    
        else:
            reward = -100


        self.result_out.put({"observation" : obs, "reward" : reward, "action" : None, "done" : True, "truncated" : False, "info" : {}})
        
    
    async def on_step(self, iteration): # on_step is a method that is called every step of the game.
        self.action = self.action_in.get()

        if iteration % 10 == 0:
            print("armybot at...", iteration)

        if self.action is None:
            print("no action returning.")
            return None
        
        time.sleep(self.tickRate)

        #Base reward
        reward = -1
        '''
        0: Train SCV
        1: Train marine
        2: Distribute workers
        '''
        # 0: Train SCV
        if self.action == 0:
            try:
                if self.can_afford(UnitTypeId.SCV):
                    for cc in self.structures(UnitTypeId.COMMANDCENTER).idle:
                        cc.train(UnitTypeId.SCV)
                        reward += 2
            except Exception as e:
                print(e)

        #1: Train marine
        elif self.action == 1:
            try:
                if self.can_afford(UnitTypeId.MARINE):
                    for bar in self.structures(UnitTypeId.BARRACKS).idle:
                        if self.can_afford(UnitTypeId.MARINE):
                            bar.train(UnitTypeId.MARINE)
                            reward += 2
            except Exception as e:
                print(e)
        
        #2: Distribute workers
        elif self.action == 2:
            for scv in self.units(UnitTypeId.SCV).idle:
                reward += 2
            await self.distribute_workers()    
            reward += 1
            print(reward)

        # Values: [MarineNr, SCVNr, Minerals, CCAvailable, BarAvailable]
        obs = np.zeros(5, dtype=np.uint16)
        obs[0] = self.army_count
        obs[1] = self.supply_workers
        obs[2] = min(self.minerals, 999)
        for cc in self.structures(UnitTypeId.COMMANDCENTER).idle:
            obs[3] += 1
        for bar in self.structures(UnitTypeId.BARRACKS).idle:
            obs[4] += 1

        #Compute reward
        try:
            if obs[2] > 50:
                reward -= (self.minerals - 50) / 100
        except Exception as e:
            print("reward",e)
            reward = 0

        truncated = False
        if iteration == 2000:
            truncated = True

        self.result_out.put({"observation" : obs, "reward" : reward, "action" : None, "done" : False, "truncated" : truncated, "info" : {}})
        