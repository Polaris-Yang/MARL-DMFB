import copy
import math
import os
import random
import sys
import time
from datetime import MINYEAR, datetime
from enum import IntEnum
from typing_extensions import runtime
from numpy.core.fromnumeric import size

from numpy.lib.function_base import select
import gym
import numpy as np
from gym import spaces, wrappers
from gym.utils import seeding
from numpy.random import poisson
from pettingzoo.utils.env import ParallelEnv
# from PIL import Image
'''
DMFBs MARL enviroment created by R.Q. Yang
'''


# action space
class Action(IntEnum):
    STALL = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
# class block and inter method


class Block:
    def __init__(self, x_min, x_max, y_min, y_max):
        if x_min > x_max or y_min > y_max:
            raise TypeError('Module() inputs are illegal')
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

    def __repr__(self):
        return 'Blocks Occupies the grid from ({},{}) to ({},{})'\
            .format(self.x_min, self.y_min, self.x_max, self.y_max)

    def isPointInside(self, point):
        ''' point is in the form of (x, y) '''
        for i in point:
            if i[0] >= self.x_min and i[0] <= self.x_max and\
                    i[1] >= self.y_min and i[1] <= self.y_max:
                return True
        else:
            return False

    def isBlockOverlap(self, m):
        if self._isLinesOverlap(self.x_min, self.x_max, m.x_min, m.x_max) and\
                self._isLinesOverlap(self.y_min, self.y_max, m.y_min, m.y_max):
            return True
        else:
            return False

    def _isLinesOverlap(self, xa_1, xa_2, xb_1, xb_2):
        if xa_1 > xb_2:
            return False
        elif xb_1 > xa_2:
            return False
        else:
            return True

# droplet class


class Droplet():
    def __init__(self, coordinate_x, coordinate_y, destination_x, destination_y):
        self.x = int(coordinate_x)
        self.y = int(coordinate_y)
        self.des_x = int(destination_x)
        self.des_y = int(destination_y)

    def __repr__(self):
        return 'Droplet is at ({},{}) and its target destination is ({},{})'.format(
            self.x, self.y, self.des_x, self.des_y)

    def __eq__(self, droplet):
        if isinstance(droplet, Droplet):
            flag = (self.x == droplet.x) and (self.y == droplet.y) and (self.des_x == droplet.des_x)\
                and (self.des_y == droplet.des_y)
            return flag
        else:
            return False

    def getDistance(self):
        return abs(self.x-self.des_x)+abs(self.y-self.des_y)

    def shfit_x(self, step):
        self.x += step

    def shfit_y(self, step):
        self.y += step

    def move(self, action, width, length):
        # width :vertical cell number, length:horizental cell number
        if action == Action.STALL:
            pass
        elif action == Action.UP:
            self.shfit_y(1)
        elif action == Action.DOWN:
            self.shfit_y(-1)
        elif action == Action.LEFT:
            self.shfit_x(-1)
        elif action == Action.RIGHT:
            self.shfit_x(1)
        else:
            raise TypeError('action is illegal')
        if self.x > length-1:
            self.x = length-1
        elif self.x < 0:
            self.x = 0
        if self.y > width-1:
            self.y = width-1
        elif self.y < 0:
            self.y = 0


class RoutingTaskManager:
    def __init__(self, width, length, n_droplets, n_blocks=0):
        self.width = width
        self.length = length
        self.n_droplets = n_droplets
        self.n_blocks = n_blocks
        self.droplets = []
        self.starts = np.zeros((self.n_droplets, 2), dtype=int)
        self.ends = np.zeros((self.n_droplets, 2), dtype=int)
        self.pasts = np.zeros((self.n_droplets,2),dtype=int)*(-5)
        self.curs = np.zeros((self.n_droplets,2),dtype=int)*(-5)
        self.distances = np.zeros((self.n_droplets,), dtype=int)
        self.blocks = []
        self.droplet_limit = int((self.width+1)*(self.length+1)/9)
        if n_droplets > self.droplet_limit:
            raise TypeError('Too many droplets for DMFB')
        self.step_count = [0] * self.n_droplets
        random.seed(datetime.now())
        self.Generate_task()

    def Generate_task(self):
        self.GenDroplets()
        self.curs = copy.deepcopy(self.starts)
        self.pasts = np.zeros((self.n_droplets,2),dtype=int)*(-5)
        self.distances = np.sum(np.abs(self.starts - self.ends), axis=1)
        self.GenRandomBlocks()

    # reset the enviorment
    def refresh(self):
        self.droplets.clear()
        self.blocks.clear()
        self.Generate_task()

    def restartforall(self):
        self.droplets.clear()
        for i in range(0, self.n_droplets):
            self.droplets.append(
                Droplet(self.starts[i][0], self.starts[i][1], self.ends[i][0], self.ends[i][1]))
        self.distances = np.sum(np.abs(self.starts - self.ends), axis=1)
        self.pasts = np.zeros((self.n_droplets,2),dtype=int)*(-5)
        self.curs = copy.deepcopy(self.starts)
        
    def resetTask(self,agent_index):
        x = np.random.randint(0, self.length, size=(2, 1))
        y = np.random.randint(0,self.width,size = (2,1))
        points = np.hstack((x, y))
        dis = abs(points[0][0]-points[1][0])+abs(points[0][1]-points[1][1])
        while dis<2:
            x = np.random.randint(0, self.length, size=(2, 1))
            y = np.random.randint(0,self.width,size = (2,1))
            points = np.hstack((x, y))
            dis = abs(points[0][0]-points[1][0])+abs(points[0][1]-points[1][1])
        start, end = points[0],points[1]
        self.droplets[agent_index] = Droplet(start[0],start[1], end[0],end[1])
        self.starts[agent_index] = start
        self.curs[agent_index] = start
        self.pasts[agent_index] = [-5,-5]
        self.ends[agent_index] = end
        self.distances[agent_index] = dis 
    
    def GenDroplets(self):
        Start_End = self._Generate_Start_End()
        self.starts = Start_End[0:self.n_droplets]
        self.ends = Start_End[self.n_droplets:]
        for i in range(0, self.n_droplets):
            self.droplets.append(
                Droplet(self.starts[i][0], self.starts[i][1], self.ends[i][0], self.ends[i][1]))

    def compute_norm_squared_EDM(self, x):
        x = x.T
        m, n = x.shape
        G = np.dot(x.T, x)
        H = np.tile(np.diag(G), (n, 1))
        return H+H.T-2*G
    
    def randomXY(self,w, l, n):
        x = np.random.randint(0, l, size=(n*2, 1))
        y = np.random.randint(0, w, size=(n*2, 1))
        Start_End = np.hstack((x, y))
        return Start_End
    
    def _Generate_Start_End(self):
        Start_End = self.randomXY(self.width, self.length, self.n_droplets)
        dis = self.compute_norm_squared_EDM(Start_End)
        strided = np.lib.stride_tricks.as_strided
        s0, s1 = dis.strides
        m = dis.shape[0]
        out = strided(dis.ravel()[1:], shape=(m-1, m),
                      strides=(s0+s1, s1)).reshape(m, -1)
        while out.min() <= 2:
            Start_End = self.randomXY(self.width, self.length, self.n_droplets)
            dis = self.compute_norm_squared_EDM(Start_End)
            s0, s1 = dis.strides
            out = strided(dis.ravel()[1:], shape=(
                m-1, m), strides=(s0+s1, s1)).reshape(m, -1)
        return Start_End

    def GenRandomBlocks(self):
        # Generate reandom blocks up to n_blocks
        if self.width < 5 or self.length < 5:
            return []
        if self.n_blocks * 4 / (self.width * self.length) > 0.2:
            print('Too many required modules in the environment.')
            return []
        self.blocks = []

        def isblocksoverlap(m, blocks):
            for mdl in blocks:
                if mdl.isBlockOverlap(m):
                    return True
            return False

        for i in range(self.n_blocks):
            x = np.random.randint(0, self.length-3)
            y = np.random.randint(0, self.width-3)
            m = Block(x, x+1, y, y+1)
            while m.isPointInside(np.vstack((self.starts, self.ends))) or isblocksoverlap(m, self.blocks):
                x = random.randrange(0, self.length - 3)
                y = random.randrange(0, self.width - 3)
                m = Block(x, x+1, y, y+1)
            self.blocks.append(m)

    def comflic_static(self, n_droplets, all_cur_position):
        static_conflic = [0] * n_droplets
        for i in range(n_droplets - 1):
            for j in range(i+1, n_droplets):
                if np.linalg.norm(all_cur_position[i]-all_cur_position[j]) < 2:
                    static_conflic[i] += 1
                    static_conflic[j] += 1
        return static_conflic

    def comflic_dynamic(self, n_droplets, all_cur_position, all_past_pisition):
        dynamic_conflict = [0] * n_droplets
        for i in range(n_droplets):
            for j in range(n_droplets):
                if i != j:
                    if np.linalg.norm(all_past_pisition[i] - all_cur_position[j]) < 2:
                        dynamic_conflict[i] += 1
                        dynamic_conflict[j] += 1
        return dynamic_conflict

    def moveDroplets(self, actions, m_health):
        if len(actions) != self.n_droplets:
            raise RuntimeError("The number of actions is not the same"
                               " as n_droplets")
        rewards = []
        pasts = []
        curs = []
        dones = self.getTaskStatus()
        for i in range(self.n_droplets):
            reward, past, cur = self.moveOneDroplet(i, actions[i], m_health)
            rewards.append(reward)
            pasts.append(past)
            curs.append(cur)
        sta = np.array(self.comflic_static(self.n_droplets, np.array(curs)))
        dy = np.array(self.comflic_dynamic(self.n_droplets, curs, pasts))
        # the nunber of obey the constraints
        constraints = np.sum(sta)+np.sum(dy)
        rewards = np.array(rewards)-2*sta-2*dy
        for i in range(self.n_droplets):
            if dones[i]:
                rewards[i] = 0.0
        if np.all(self.getTaskStatus()) == True:
            rewards = [i+20 for i in rewards]
        rewards = list(rewards)
        position_change = list(
            np.all((np.array(curs) == np.array(pasts)), axis=1))
        return rewards, constraints, position_change

    def _isTouchingBlocks(self, point):
        for m in self.blocks:
            if point[0] >= m.x_min and\
                    point[0] <= m.x_max and\
                    point[1] >= m.y_min and\
                    point[1] <= m.y_max:
                return True
        return False

    def _isinvalidaction(self):
        position = np.zeros((self.n_droplets, 2))
        for i, d in enumerate(self.droplets):
            position[i][0], position[i][1] = d.x, d.y
        dis = self.compute_norm_squared_EDM(position)
        m = dis.shape[0]
        strided = np.lib.stride_tricks.as_strided
        s0, s1 = dis.strides
        out = strided(dis.ravel()[1:], shape=(m-1, m),
                      strides=(s0+s1, s1)).reshape(m, -1)
        if out.min() == 0:
            return True
        else:
            return False

    def moveOneDroplet(self, droplet_index, action, m_health, b_multithread=False):
        if droplet_index >= self.n_droplets:
            raise RuntimeError(
                "The droplet index {} is out of bound".format(droplet_index))
        x = self.droplets[droplet_index].x
        y = self.droplets[droplet_index].y
        i = droplet_index
        if b_multithread:
            while self._waitOtherActions(i):
                pass
            self.step_count[i] += 1

        if self.distances[i] == 0:
            reward = 0.0
        else:
            prob = self.getMoveProb(self.droplets[droplet_index], m_health)
            if random.random() <= prob:
                self.droplets[droplet_index].move(
                    action, self.width, self.length)
                if self._isTouchingBlocks([self.droplets[droplet_index].x, self.droplets[droplet_index].y]):
                    self.droplets[droplet_index].x = x
                    self.droplets[droplet_index].y = y
                if self._isinvalidaction():
                    self.droplets[droplet_index].x = x
                    self.droplets[droplet_index].y = y
            new_dist = self.droplets[droplet_index].getDistance()
            if new_dist == self.distances[droplet_index] and self.distances[droplet_index] == 0:
                reward = 0.0
            # stall in past pisition
            elif new_dist == self.distances[droplet_index]:
                reward = -0.25
            # closer to the destination
            elif new_dist < self.distances[droplet_index]:
                reward = -0.1
            else:
                reward = -0.4  # penalty for taking one more step
            self.distances[droplet_index] = new_dist
        self.pasts[droplet_index] = np.array([x, y])
        past = np.array([x, y])
        cur = np.array([self.droplets[droplet_index].x,
                       self.droplets[droplet_index].y])
        self.curs[droplet_index] = copy.deepcopy(cur)
        return reward, past, cur

    def _waitOtherActions(self, index):
        for i, count in enumerate(self.step_count):
            if i == index:
                continue
            if self.step_count[i] < count:
                return True
        return False

    def getMoveProb(self, droplet, m_health):
        prob = m_health[droplet.y][droplet.x]
        return prob

    def getTaskStatus(self):
        return [i == 0 for i in self.distances]


class DMFBenv(ParallelEnv):
    """ A DMFB biochip environment
    [0,0]
        +---l---+-> x
        w       |
        +-------+
        |     [1,2]
        V
        y
    """
    metadata = {'render.modes': ['human', 'rgb_array']}
    # 环境初始化

    def __init__(self, width, length, n_agents, n_blocks=0, b_degrade=False, per_degrade=0.5):
        super(DMFBenv, self).__init__()
        assert width >= 5 and length >= 5
        assert n_agents > 0
        self.actions = Action
        self.agents = ["player_{}".format(i) for i in range(n_agents)]
        self.possible_agents = self.agents[:]
        self.action_spaces = {name: spaces.Discrete(len(self.actions))
                              for name in self.agents}
        self.observation_spaces = {name: spaces.Box(
            low=0, high=1, shape=(width, length, 3), dtype='uint8')
            for name in self.agents}
        self.rewards = {i: 0. for i in self.agents}
        self.dones = {i: False for i in self.agents}
        self.width = width
        self.length = length
        self.b_degrade = b_degrade
        self.routing_manager = RoutingTaskManager(
            width, length, n_agents, n_blocks)
        self.max_step = (width + length)*2
        self.m_health = np.ones((width, length))
        self.m_usage = np.zeros((width, length))
        self.constraints = 0
        if b_degrade:
            self.m_degrade = np.random.rand(width, length)
            self.m_degrade = self.m_degrade * 0.4 + 0.6
            selection = np.random.rand(width, length)
            per_healthy = 1. - per_degrade
            self.m_degrade[selection < per_healthy] = 1.0
        else:
            self.m_degrade = np.ones((width, length))
        # variables below change every game
        self.step_count = 0

    def step(self, actions):
        self.step_count += 1
        success = 0
        if isinstance(actions, dict):
            acts = [actions[agent] for agent in self.agents]
        elif isinstance(actions, list):
            acts = actions
        else:
            raise TypeError('wrong actions')
        rewards, constraints, p_c = self.routing_manager.moveDroplets(
            acts, self.m_health)
        self.constraints += constraints
        for key, r in zip(self.agents, rewards):
            self.rewards[key] = r
        obs = self.getObs()  # patitial observed consist of the Obs
        if self.step_count < self.max_step:
            status = self.routing_manager.getTaskStatus()
            if np.all(status) and self.constraints == 0:
                success = 1
            for key, s in zip(self.agents, status):
                self.dones[key] = s
            self.addUsage(p_c)
        else:
            for key in self.agents:
                self.dones[key] = True
        info = {'constraints': constraints, 'success': success}
        return obs, self.rewards, self.dones, info

    def reset(self):
        self.rewards = {i: 0 for i in self.agents}
        self.dones = {i: False for i in self.agents}
        self.step_count = 0
        self.constraints = 0
        self.routing_manager.refresh()
        self.updateHealth()
        obs = self.getObs()
        return obs

    def restart(self, index=None):
        self.routing_manager.restartforall()
        self.rewards = {i: 0.0 for i in self.agents}
        self.dones = {i: False for i in self.agents}
        self.step_count = 0
        self.constraints = 0
        if index:
            return self.getOneObs(index)
        else:
            return self.getObs()

    def seed(self, seed=None):
        pass

    def close(self):
        pass

    def addUsage(self, is_state_change):
        done = self.routing_manager.getTaskStatus()
        for i in range(self.routing_manager.n_droplets):
            if not done[i]:
                self.m_usage[self.routing_manager.droplets[i].y][self.routing_manager.droplets[i].x] += 1

    def updateHealth(self):
        if not self.b_degrade:
            return
        index = self.m_usage > 50.0  # degrade here
        self.m_health[index] = self.m_health[index] * self.m_degrade[index]
        self.m_usage[index] = 0

    def getOneObs(self, agent_index):
        """
        RGB format of image
        Obstacles - red in layer 0
        Goal      - greed in layer 1
        Droplet   - blue in layer 2
        """
        obs = np.zeros(shape=(self.width, self.length, 3))
        # First add other droplets in 0 layer
        for j in range(self.routing_manager.n_droplets):
            if j == agent_index:
                continue
            o_drp = self.routing_manager.droplets[j]
            obs[o_drp.y][o_drp.x][0] = 1
        # Add destination in 1 layer
        o_drp = self.routing_manager.droplets[agent_index]
        # print(o_drp)
        obs[o_drp.des_y][o_drp.des_x][1] = 1
        # Add droplet in 2 layer
        obs[o_drp.y][o_drp.x][2] = 1
        return obs

    def getObs(self):
        observations = {}
        for i, agent in enumerate(self.agents):
            observations[agent] = self.getOneObs(i)
        return observations

    def render(self, mode='human', close=False):
        pass


