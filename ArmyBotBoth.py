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
        reward = 0
        
        # Values: [MarineHealth, ZergDist, WeaponCD, MarineNr, SCVNr, Minerals, CCAvailable, BarAvailable]
        obs = np.zeros(8, dtype=np.uint16)
        obs[1] = 0
        obs[3] = self.army_count
        obs[4] = self.supply_workers
        obs[5] = self.minerals
        for cc in self.structures(UnitTypeId.COMMANDCENTER).idle:
            obs[6] += 1
        for bar in self.structures(UnitTypeId.BARRACKS).idle:
            obs[7] += 1
        
        if str(game_result) == "Result.Victory":
            for marine in self.units(UnitTypeId.MARINE):
                if marine.health_percentage < 1:
                    reward += marine.health
                    obs[0] = marine.health
                    obs[2] = int(marine.weapon_cooldown * 100)
                    
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
        0: Force Move
        1: Attack Move
        2: Train SCV
        3: Train marine
        4: Distribute workers
        '''
        # 0: Force Move
        if self.action == 0:
            try:
                bar = (random.choice(self.structures(UnitTypeId.BARRACKS)))
                marine = self.units(UnitTypeId.MARINE)
                marine.furthest_to(bar).move(random.choice(self.structures(UnitTypeId.MISSILETURRET)))
            except Exception as e:
                print(e)

        #1: Attack Move
        elif self.action == 1:
            try:
                if self.enemy_units:
                    zergling = random.choice(self.enemy_units)
                    bar = (random.choice(self.structures(UnitTypeId.BARRACKS)))
                    marine = self.units(UnitTypeId.MARINE)
                    marine.furthest_to(bar).attack(zergling)
            except Exception as e:
                print(e)

        # 2: Train SCV
        if self.action == 2:
            try:
                if self.can_afford(UnitTypeId.SCV):
                    for cc in self.structures(UnitTypeId.COMMANDCENTER).idle:
                        cc.train(UnitTypeId.SCV)
                        reward += 2
            except Exception as e:
                print(e)

        #3: Train marine
        elif self.action == 3:
            try:
                if self.can_afford(UnitTypeId.MARINE):
                    for bar in self.structures(UnitTypeId.BARRACKS).idle:
                        if self.can_afford(UnitTypeId.MARINE):
                            bar.train(UnitTypeId.MARINE)
                            reward += 2
            except Exception as e:
                print(e)
        
        #4: Distribute workers
        elif self.action == 4:
            for scv in self.units(UnitTypeId.SCV).idle:
                reward += 2
            await self.distribute_workers()    

        # Values: [MarineHealth, ZergDist, WeaponCD, MarineNr, SCVNr, Minerals, CCAvailable, BarAvailable]
        obs = np.zeros(8, dtype=np.uint16)
        obs[3] = self.army_count
        obs[4] = self.supply_workers
        obs[5] = self.minerals
        for cc in self.structures(UnitTypeId.COMMANDCENTER).idle:
            obs[6] += 1
        for bar in self.structures(UnitTypeId.BARRACKS).idle:
            obs[7] += 1

        try:
            for scv in self.units(UnitTypeId.SCV).idle:
                reward -= 2
            if obs[2] > 50:
                reward -= (self.minerals - 50) / 100


            # iterate through our marines:
            bar = (random.choice(self.structures(UnitTypeId.BARRACKS)))
            marine = self.units(UnitTypeId.MARINE)
            furthest_marine = marine.furthest_to(bar)

            obs[0] = furthest_marine.health
            if self.enemy_units:
                zergling = random.choice(self.enemy_units)
                obs[1] = int(furthest_marine.distance_to(zergling) * 10)
            else:
                obs[1] = 0

            if furthest_marine.weapon_cooldown > 0:
                obs[2] = 1

            # if marine is attacking and is in range of enemy unit:
            if furthest_marine.is_attacking:
                reward += 1
                if self.enemy_units.closer_than(5, furthest_marine) and furthest_marine.weapon_cooldown == 0:
                    reward += 1
            else:
                if furthest_marine.weapon_cooldown > 0:
                    reward += 2

        except Exception as e:
            print("reward",e)
            reward = 0

        truncated = False
        if iteration == 2000:
            truncated = True

        self.result_out.put({"observation" : obs, "reward" : reward, "action" : None, "done" : False, "truncated" : truncated, "info" : {}})