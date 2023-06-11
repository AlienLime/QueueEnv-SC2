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
        self.action_in = action_in
        self.result_out = result_out

    async def on_end(self,game_result):
        print ("Game over!")
        reward = 0
        
        # Values: [MarineHealth, ZergDist, WeaponCD]
        obs = np.zeros(3, dtype=np.uint16)

        if str(game_result) == "Result.Victory":
            for marine in self.units(UnitTypeId.MARINE):
                if marine.health_percentage < 1:
                    reward += (marine.health * marine.health) / 5
                    # Set obs[0]
                    obs[0] = marine.health
                    
                    # Set obs[2]
                    if marine.weapon_cooldown > 0:
                        obs[2] = 1

                    
        else:
            reward = 0


        self.result_out.put({"observation" : obs, "reward" : reward, "action" : None, "done" : True, "truncated" : False, "info" : {}})
        
    
    async def on_step(self, iteration): # on_step is a method that is called every step of the game.
        self.action = self.action_in.get()

        if self.action is None:
            print("no action returning.")
            return None
        
        '''
        0: Force Move
        1: Attack Move
        '''
        # 0: Force Move
        if self.action == 0:
            try:
                for marine in self.units(UnitTypeId.MARINE):
                    for mt in self.structures(UnitTypeId.MISSILETURRET):
                        marine.move(mt)
            except Exception as e:
                print(e)

        #1: Attack Move
        elif self.action == 1:
            try:
                for marine in self.units(UnitTypeId.MARINE):
                    marine.attack(random.choice(self.enemy_units))
            except Exception as e:
                print(e)

        # Values: [MarineHealth, ZergDist, WeaponCD, MarineNr, SCVNr, Minerals]
        obs = np.zeros(3, dtype=np.uint16)

        try:
            # iterate through our marines:
            for marine in self.units(UnitTypeId.MARINE):
                # Set obs[0]
                obs[0] = marine.health

                # Set obs[1]
                if self.enemy_units:
                    zergling = random.choice(self.enemy_units)
                    obs[1] = int(marine.distance_to(zergling) * 10)
                
                # Set obs[2]
                if marine.weapon_cooldown > 0:
                    obs[2] = 1

                # Compute reward
                reward = -1

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

        self.result_out.put({"observation" : obs, "reward" : reward, "action" : None, "done" : False, "truncated" : False, "info" : {}})