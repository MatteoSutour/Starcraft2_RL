from sc2.bot_ai import BotAI  # parent class we inherit from
from sc2.data import Difficulty, Race  # difficulty for bots, race for the 1 of 3 races
from sc2.main import run_game  # function that facilitates actually running the agents in games
from sc2.player import Bot, Computer  # wrapper for whether or not the agent is one of your bots, or a "computer" player
from sc2 import maps  # maps method for loading maps to play in.
from loguru import logger

from sc2.ids.unit_typeid import UnitTypeId
import random
import cv2
import math
import numpy as np
import sys
import pickle
import time
from enum import Enum

SAVE_REPLAY = False

tag_mapping = {}
total_steps = 10000
steps_for_pun = np.linspace(0, 1, total_steps)
step_punishment = ((np.exp(steps_for_pun ** 3) / 10) - 0.1) * 10

class UnitName(Enum):
    Zealot = 0
    Stalker = 1


class IncrediBot(BotAI):  # inhereits from BotAI
    async def on_step(self, iteration: int):  # on_step is a method that is called every step of the game.
        if tag_mapping == {}:
            for ind, unit in enumerate(self.units):
                tag_mapping[unit.tag] = ind
            for ind, enemy_unit in enumerate(self.enemy_units):
                tag_mapping[enemy_unit.tag] = ind + 7

        no_action = True

        enemy_health_base = 0
        for enemy_unit in self.enemy_units:
            enemy_health_base += enemy_unit.health

        while no_action:
            # print('No action')
            try:
                with open('state_rwd_action.pkl', 'rb') as f:
                    state_rwd_action = pickle.load(f)

                    if state_rwd_action['action'] is None:
                        # print("No action yet")
                        no_action = True
                    else:
                        # print("Action found")
                        no_action = False
            except:
                pass

        action = state_rwd_action['action']
        for unit in self.units:  # define strategie for each units
            # print('iter unit')
            # print("Action found")
            action_ind = action[tag_mapping[unit.tag]]
            # 0 : move to right
            if action_ind == 0:
                unit.move(unit.position + (3, 0))

            # 1: move to left
            elif action_ind == 1:
                unit.move(unit.position + (-3, 0))

            # 2: move up
            elif action_ind == 2:
                unit.move(unit.position + (0, 3))

            # 3: move down
            elif action_ind == 3:
                unit.move(unit.position + (0, -3))

            # 4: attack the closest enemie
            elif action_ind == 4:
                if self.enemy_units:
                    try:
                        unit.attack(self.enemy_units.closest_to(unit))
                    except Exception as e:
                        print('No close enemy unit')

        map = np.zeros((self.game_info.map_size[0], self.game_info.map_size[1], 3), dtype=np.uint8)
        state = np.zeros((16, 5), dtype=np.uint8)
        enemy_health_after = 0
        # draw our units (with health):
        try:
            for unit in self.units:
                pos = unit.position
                c = [255, 75, 75]
                # get health:
                fraction = unit.health / unit.health_max if unit.health_max > 0 else 0.0001
                map[math.ceil(pos.y)][math.ceil(pos.x)] = [int(fraction * i) for i in c]
                state[tag_mapping[unit.tag]] = np.array([pos.x, pos.y, unit.health, 0, UnitName[unit.name.replace('_RL','')].value])  # The 0 is for the team
        except Exception as e:
            print(e)

        # draw the enemy units (with health):
        for enemy_unit in self.enemy_units:
            enemy_health_after += enemy_unit.health
            pos = enemy_unit.position
            c = [100, 0, 255]
            # get unit health fraction:
            fraction = enemy_unit.health / enemy_unit.health_max if enemy_unit.health_max > 0 else 0.0001
            map[math.ceil(pos.y)][math.ceil(pos.x)] = [int(fraction * i) for i in c]
            state[tag_mapping[enemy_unit.tag]] = np.array([pos.x, pos.y, enemy_unit.health, 1, UnitName[enemy_unit.name].value])  # The 1 is for the team

        # show map with opencv, resized to be larger:
        # horizontal flip:

        cv2.imshow('map', cv2.flip(cv2.resize(map, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST), 0))
        cv2.waitKey(100)

        if SAVE_REPLAY:
            # save map image into "replays dir"
            cv2.imwrite(f"replays/{int(time.time())}-{iteration}.png", map)

        reward = enemy_health_base - enemy_health_after

        if iteration % 20 == 0:
            print(f"Iter: {iteration}. RWD: {reward}.")

        # write the file:
        data = {"state": state, "reward": reward, "action": None,
                "done": False}  # empty action waiting for the next one!

        with open('state_rwd_action.pkl', 'wb') as f:
            pickle.dump(data, f)
        # print('not self unit : ', not self.units)

        if not self.units:
            print('D')
            rwd = -1000

            with open("results.txt", "a") as f:
                f.write("D\n")
            state = np.zeros((16, 5), dtype=np.uint8)
            data = {"state": state, "reward": rwd, "action": None, "done": True}  # empty action waiting for the next one!
            with open('state_rwd_action.pkl', 'wb') as f:
                pickle.dump(data, f)

            cv2.destroyAllWindows()
            cv2.waitKey(1)
            time.sleep(3)
            sys.exit()

        if not self.enemy_units:
            print('V')
            rwd = 1000

            with open("results.txt", "a") as f:
                f.write("V\n")
            state = np.zeros((16, 5), dtype=np.uint8)
            data = {"state": state, "reward": rwd, "action": None, "done": True}  # empty action waiting for the next one!
            with open('state_rwd_action.pkl', 'wb') as f:
                pickle.dump(data, f)
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            time.sleep(3)
            sys.exit()


result = run_game(  # run_game is a function that runs the game.
    maps.get("3s5z"),  # the map we are playing on
    [Bot(Race.Protoss, IncrediBot()),  # runs our coded bot, protoss race, and we pass our bot object
     Computer(Race.Protoss, Difficulty.Easy)],  # runs a pre-made computer agent, zerg race, with a hard difficulty.
    realtime=False,  # When set to True, the agent is limited in how long each step can take to process.
)
print("result ok")
