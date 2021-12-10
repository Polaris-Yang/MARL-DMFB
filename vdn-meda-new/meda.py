import copy
import math
import queue
import random
import numpy as np
from PIL import Image
from enum import IntEnum
from datetime import datetime
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from pettingzoo.utils.env import ParallelEnv

"""
New MEDA Environment for the MARL work
Use PettingZoo as the envrionment wrapper
2020/12/12 by T.-C. Liang
"""


class Action(IntEnum):
    N = 0  # North
    E = 1  # East
    S = 2  # South
    W = 3  # West
    NE = 4
    SE = 5
    SW = 6
    NW = 7


class Droplet:
    def __init__(self, x_min, x_max, y_min, y_max):
        if x_min > x_max or y_min > y_max:
            raise TypeError('Droplet() inputs are illegal')
        if x_max - x_min != y_max - y_min:
            raise RuntimeError('Droplet() is not a square')
        self.x_min = int(x_min)
        self.x_max = int(x_max)
        self.y_min = int(y_min)
        self.y_max = int(y_max)
        self.x_center = int((x_min + x_max) / 2)
        self.y_center = int((y_min + y_max) / 2)
        self.radius = x_max - self.x_center

    def __repr__(self):
        return "Droplet(x = " + str(self.x_center) + ", y = " +\
            str(self.y_center) + ", r = " +\
            str(self.radius) + ")"

    def __eq__(self, rhs):
        if isinstance(rhs, Droplet):
            return self.x_min == rhs.x_min and self.x_max == rhs.x_max and\
                self.y_min == rhs.y_min and self.y_max == rhs.y_max and\
                self.x_center == rhs.x_center and\
                self.y_center == rhs.y_center
        else:
            return False

    def isPointInside(self, point):
        ''' point is in the form of (y, x) '''
        if point[0] >= self.y_min and point[0] <= self.y_max and\
                point[1] >= self.x_min and point[1] <= self.x_max:
            return True
        else:
            return False

    def isDropletOverlap(self, m):
        if self._isLinesOverlap(self.x_min, self.x_max, m.x_min, m.x_max) and\
                self._isLinesOverlap(self.y_min, self.y_max, m.y_min, m.y_max):
            return True
        else:
            return False

    def isTooClose(self, m):
        """ Return true if distance is less than 1.5 radius sum """
        distance = self.getDistance(m)
        return distance < 1.5 * (self.radius + m.radius + 2)

    def _isLinesOverlap(self, xa_1, xa_2, xb_1, xb_2):
        if xa_1 > xb_2:
            return False
        elif xb_1 > xa_2:
            return False
        else:
            return True

    def getDistance(self, droplet):
        delta_x = self.x_center - droplet.x_center
        delta_y = self.y_center - droplet.y_center
        return math.sqrt(delta_x * delta_x + delta_y * delta_y)

    def shiftX(self, step):
        self.x_min += step
        self.x_max += step
        self.x_center += step

    def shiftY(self, step):
        self.y_min += step
        self.y_max += step
        self.y_center += step

    def move(self, action, width, length):
        if action == Action.N:
            self.shiftY(-3)
        elif action == Action.E:
            self.shiftX(3)
        elif action == Action.S:
            self.shiftY(3)
        elif action == Action.W:
            self.shiftX(-3)
        elif action == Action.NE:
            self.shiftX(2)
            self.shiftY(-2)
        elif action == Action.SE:
            self.shiftX(2)
            self.shiftY(2)
        elif action == Action.SW:
            self.shiftX(-2)
            self.shiftY(2)
        elif action == Action.NW:
            self.shiftX(-2)
            self.shiftY(-2)
        if self.x_max >= length:
            self.shiftX(length - 1 - self.x_max)
        elif self.x_min < 0:
            self.shiftX(0 - self.x_min)
        if self.y_max >= width:
            self.shiftY(width - 1 - self.y_max)
        elif self.y_min < 0:
            self.shiftY(0 - self.y_min)


class RoutingTaskManager:
    def __init__(self, w, l, n_droplets):
        self.width = w
        self.length = l
        self.n_droplets = n_droplets
        self.starts = []
        self.droplets = []
        self.destinations = []
        self.distances = []
        self.n_limit = int(w / 15) * int(l / 15)
        if n_droplets > self.n_limit:
            raise RuntimeError("Too many droplets in the " + str(w) + "x" +
                               str(l) + " MEDA array")
        random.seed(datetime.now())
        for i in range(n_droplets):
            self.addTask()
        self.step_count = [0] * self.n_droplets

    def refresh(self):
        self.starts.clear()
        self.droplets.clear()
        self.destinations.clear()
        self.distances.clear()
        for i in range(self.n_droplets):
            self.addTask()

    def restart(self):
        self.droplets = copy.deepcopy(self.starts)
        self._updateDistances()

    def addTask(self):
        if len(self.droplets) >= self.n_limit:
            return
        self._genLegalDroplet(self.droplets)
        self._genLegalDroplet(self.destinations)
        while(self.droplets[-1].isDropletOverlap(self.destinations[-1])):
            self.destinations.pop()
            self._genLegalDroplet(self.destinations)
        self.distances.append(
            self.droplets[-1].getDistance(self.destinations[-1]))
        self.starts.append(copy.deepcopy(self.droplets[-1]))

    def resetTask(self, agent_index):
        drp_center = self.getRandomYX()
        dst_center = self.getRandomYX()
        while dst_center == drp_center or\
                self._isCenterTooClose(dst_center, self.destinations):
            dst_center = self.getRandomYX()
        drp = self._getDropletFromCenter(drp_center)
        dst = self._getDropletFromCenter(dst_center)
        self.droplets[agent_index] = drp
        self.destinations[agent_index] = dst
        self.distances[agent_index] = drp.getDistance(dst)
        self.starts[agent_index] = copy.deepcopy(drp)

    def _isCenterTooClose(self, center, l_droplets, r=3):
        for d in l_droplets:
            if abs(center[1] - d.x_center) <= 2*r and\
                    abs(center[0] - d.y_center) <= 2*r:
                return True
        return False

    def _getDropletFromCenter(self, center, r=3):
        return Droplet(center[1] - r, center[1] + r, center[0] - r,
                       center[0] + r)

    def _genLegalDroplet(self, dtype):
        d_center = self.getRandomYX()
        new_d = Droplet(d_center[1] - 3, d_center[1] + 3, d_center[0] - 3,
                        d_center[0] + 3)
        while(not self._isGoodDroplet(new_d, dtype)):
            d_center = self.getRandomYX()
            new_d = Droplet(d_center[1] - 3, d_center[1] + 3, d_center[0] - 3,
                            d_center[0] + 3)
        dtype.append(new_d)

    def getRandomYX(self):
        return (random.randint(3, self.width - 4),
                random.randint(3, self.length - 4))

    def _isGoodDroplet(self, new_d, dtype):
        for d in dtype:
            if(d.isTooClose(new_d)):
                return False
        return True

    def _updateDistances(self):
        dist = []
        for drp, dst in zip(self.droplets, self.destinations):
            dist.append(drp.getDistance(dst))
        self.distances = dist

    def moveDroplets(self, actions, m_health):
        if len(actions) != self.n_droplets:
            raise RuntimeError("The number of actions is not the same"
                               " as n_droplets")
        rewards = []
        fail = 0
        for i in range(self.n_droplets):
            rewards.append(self.moveOneDroplet(i, actions[i], m_health))
        if self.isMixing():
            fail = 1
        punish = self.calPunish()
        for i in range(len(rewards)):
            rewards[i] += punish[i]
        return rewards, fail

    def moveOneDroplet(self, droplet_index, action, m_health,
                       b_multithread=False):
        """ Used for multi-threads """
        if not droplet_index < self.n_droplets:
            raise RuntimeError("The droplet index {} is out of bound"
                               .format(droplet_index))
        i = droplet_index
        if b_multithread:
            while self._waitOtherActions(i):
                pass
            self.step_count[i] += 1
        goal_dist = self.droplets[i].radius + self.destinations[i].radius
        if self.distances[i] < goal_dist:  # already achieved goal
            self.droplets[i] = copy.deepcopy(self.destinations[i])
            self.distances[i] = 0
            reward = 0.0
        else:
            prob = self.getMoveProb(self.droplets[i], m_health)
            if random.random() <= prob:
                self.droplets[i].move(action, self.width, self.length)
            new_dist = self.droplets[i].getDistance(self.destinations[i])
            if new_dist < goal_dist:  # get to the destination
                reward = 1.0
            elif new_dist < self.distances[i]:  # closer to the destination
                reward = -0.05
            else:
                reward = -0.1  # penalty for taking one more step
            self.distances[i] = new_dist
        return reward

    def _waitOtherActions(self, index):
        for i, count in enumerate(self.step_count):
            if i == index:
                continue
            if self.step_count[i] < count:
                return True
        return False

    def getMoveProb(self, droplet, m_health):
        count = 0
        prob = 0.0
        for y in range(droplet.y_min, droplet.y_max + 1):
            for x in range(droplet.x_min, droplet.x_max + 1):
                prob += m_health[y][x]
                count += 1
        return prob / float(count)

    def tooCloseToOthers(self, index):
        for i in range(self.n_droplets):
            if i == index:
                continue
            safe_dst = self.droplets[index].radius + self.droplets[i].radius
            real_dst = self.droplets[index].getDistance(self.droplets[i])
            if real_dst < 1.5 * safe_dst:
                return True
        return False

    def calPunish(self):
        punish = [0]*self.n_droplets
        for i in range(self.n_droplets-1):
            for j in range(i+1, self.n_droplets):
                safe_dst = self.droplets[i].radius + self.droplets[j].radius
                real_dst = self.droplets[i].getDistance(self.droplets[j])
                if real_dst < 1.5 * safe_dst:
                    punish[i] -= 0.8
                    punish[j] -= 0.8
        return punish

    def isMixing(self):
        for i in range(self.n_droplets-1):
            for j in range(i+1, self.n_droplets):
                safe_dst = self.droplets[i].radius + self.droplets[j].radius
                real_dst = self.droplets[i].getDistance(self.droplets[j])
                if real_dst <= safe_dst:
                    return True
        return False

    def getTaskStatus(self):
        goal_distances = []
        for drp, dst in zip(self.droplets, self.destinations):
            goal_distances.append(drp.radius + dst.radius)
        return [dist < gd for dist, gd in zip(self.distances, goal_distances)]


class BaseLineRouter:
    def __init__(self, w, l):
        self.width = w
        self.length = l

    def getEstimatedReward(self, routing_manager, m_health=None):
        road_map = []
        trajectories = []
        max_step = 0
        for drp, dest in zip(routing_manager.droplets,
                             routing_manager.destinations):
            actions = self.addPath(road_map, drp, dest)
            trajectories.append(actions)
            if len(actions) > max_step:
                max_step = len(actions)
        for i, actions in enumerate(trajectories):
            l = len(actions)
            if l < max_step:
                trajectories[i] += [Action.N] * (max_step - l)
        rewards = []
        steps = np.array([0. for d in routing_manager.droplets])
        for i in range(max_step):
            actions_by_droplets = [actions[i] for actions in trajectories]
            d_rewards = routing_manager.moveDroplets(actions_by_droplets,
                                                     np.ones((self.width, self.length)))
            np_r = np.average(d_rewards)
            if m_health is None:
                rewards.append(np_r)
            else:
                move_probs = np.array(
                    [routing_manager.getMoveProb(drp, m_health)
                     for drp in routing_manager.droplets])
                fail_probs = 1. - move_probs
                discount_r = np_r * move_probs - 0.9 * fail_probs * move_probs\
                    - 1.8 * fail_probs * fail_probs * move_probs
                rewards.append(np.nanmean(discount_r))
                steps = steps + 1. / move_probs
        routing_manager.restart()  # this is important to revert the game
        if m_health is None:
            return sum(rewards), max_step
        else:
            return sum(rewards), max(steps)

    def markLocation(self, road_map, drp, value):
        for y in range(drp.y_min, drp.y_max + 1):
            for x in range(drp.x_min, drp.x_max + 1):
                road_map[y][x] = value

    def addPath(self, road_map, drp, dest):
        actions = []
        delta_x = dest.x_center - drp.x_center
        delta_y = dest.y_center - drp.y_center
        if delta_x > 0:
            x_moves = [Action.E] * int(delta_x / 3)
        else:
            x_moves = [Action.W] * int(abs(delta_x) / 3)
        if delta_y > 0:
            y_moves = [Action.S] * int(delta_y / 3)
        else:
            y_moves = [Action.N] * int(abs(delta_y) / 3)
        for i in range(len(x_moves)):
            path = x_moves[:i] + y_moves + x_moves[i:]
            valid_path = True
            temp_drp = copy.deepcopy(drp)
            for j, act in enumerate(path):
                next_drp = copy.deepcopy(temp_drp)
                next_drp.move(act, self.width, self.length)
                if self.checkValidMove(next_drp, temp_drp, road_map, j + 1):
                    temp_drp = next_drp
                else:
                    valid_path = False
                    break
            if valid_path:
                actions = path
                break
        if len(actions) == 0:
            if len(y_moves) > 0:
                i = random.choice(range(len(y_moves)))
                action = y_moves[:i] + x_moves + y_moves[i:]
            else:
                action = x_moves
        this_map = np.full((self.width, self.length), -1)
        move_drp = copy.deepcopy(drp)
        for step, act in enumerate(actions):
            self.markLocation(this_map, move_drp, step)
            move_drp.move(act, self.width, self.length)
        self.markLocation(this_map, move_drp, len(actions))
        road_map.append(this_map)
        return actions

    def getScanArea(self, next_drp, prev_drp):
        points = set([])
        for y in range(next_drp.y_min, next_drp.y_max + 1):
            for x in range(next_drp.x_min, next_drp.x_max + 1):
                points.add((y, x))
        for y in range(prev_drp.y_min, prev_drp.y_max + 1):
            for x in range(prev_drp.x_min, prev_drp.x_max + 1):
                points.discard((y, x))
        return list(points)

    def checkValidMove(self, next_drp, prev_drp, road_map, next_v):
        scan_area = self.getScanArea(next_drp, prev_drp)
        for y, x in scan_area:
            for r_map in road_map:
                if r_map[y][x] >= next_v - 1 and r_map[y][x] <= next_v + 1:
                    return False
        return True


class MEDAEnv(ParallelEnv):
    """ A MEDA biochip environment
        [0,0]
          +---l---+-> x
          w       |
          +-------+
          |     [1,2]
          V
          y
    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, w, l, n_agents, b_degrade=False,
                 per_degrade=0.1):
        super(MEDAEnv, self).__init__()
        assert w > 0 and l > 0
        assert n_agents > 0
        # PettingZoo Gym setup
        self.actions = Action
        self.agents = ["player_{}".format(i) for i in range(n_agents)]
        self.possible_agents = self.agents[:]
        self.action_spaces = {name: spaces.Discrete(len(self.actions))
                              for name in self.agents}
        self.observation_spaces = {name: spaces.Box(
            low=0, high=1, shape=(3, w, l), dtype='uint8')
            for name in self.agents}
        #self.reward_range = (-1.0, 1.00)
        self.rewards = {i: 0. for i in self.agents}
        self.dones = {i: False for i in self.agents}
        # Other data members
        self.width = w
        self.length = l
        self.routing_manager = RoutingTaskManager(w, l, n_agents)
        self.b_degrade = b_degrade
        self.max_step = (w + l)
        self.fails = 0
        self.m_health = np.ones((w, l))
        self.m_usage = np.zeros((w, l))
        self.step_count = 0
        if b_degrade:
            self.m_degrade = np.random.rand(w, l)
            self.m_degrade = self.m_degrade * 0.4 + 0.6
            selection = np.random.rand(w, l)
            per_healthy = 1. - per_degrade
            self.m_degrade[selection < per_healthy] = 1.0
        else:
            self.m_degrade = np.ones((w, l))
        # variables below change every game

    def step(self, actions):
        self.step_count += 1
        success = 0
        if isinstance(actions, dict):
            acts = [actions[agent] for agent in self.agents]
        elif isinstance(actions, list):
            acts = actions
        rewards, fail = self.routing_manager.moveDroplets(acts, self.m_health)
        self.fails += fail
        for key, r in zip(self.agents, rewards):
            self.rewards[key] = r
        obs = self.getObs()
        if self.step_count < self.max_step:
            status = self.routing_manager.getTaskStatus()
            if np.all(status) and self.fails == 0:
                success = 1
            for key, s in zip(self.agents, status):
                self.dones[key] = s
            self.addUsage()
        else:
            for key in self.agents:
                self.dones[key] = True
        return obs, self.rewards, self.dones, {'success': success}

    def reset(self):
        self.rewards = {i: 0. for i in self.agents}
        self.dones = {i: False for i in self.agents}
        self.step_count = 0
        self.fails = 0
        self.routing_manager.refresh()
        obs = self.getObs()
        self.updateHealth()
        return obs

    def restart(self, index=None):
        """ Used for evaluation """
        self.routing_manager.restart()
        self.rewards = {i: 0. for i in self.agents}
        self.dones = {i: False for i in self.agents}
        self.step_count = 0
        if index:
            return self.getOneObs(index)
        else:
            return self.getObs()

    def render(self, mode='human'):
        """ Show environment """
        # goal:2, pos:1, degrade: -1
        if mode == 'human':
            img = np.zeros(shape=(self.width, self.length))
            for drp, des in zip(self.routing_manager.droplets,
                                self.routing_manager.destinations):
                for y in range(des.y_min, des.y_max + 1):
                    for x in range(des.x_min, des.x_max + 1):
                        img[y][x] = 2
                for y in range(drp.y_min, drp.y_max + 1):
                    for x in range(drp.x_min, drp.x_max + 1):
                        img[y][x] = 1
            if self.b_degrade:
                img[self.m_health < 0.5] = -1
            return img
        elif mode == 'rgb_array':
            # Default is grey
            img = np.full((self.width, self.length, 3), 192, dtype=np.uint8)
            if self.b_degrade:
                # img[self.m_health < 0.7] = [255, 102, 255] #light purple
                # img[self.m_health < 0.5] = [255, 153, 255] #purple
                img[self.m_health < 0.7] = [128, 128, 128]  # light grey
                img[self.m_health < 0.5] = [64, 64, 64]  # grey
            for des in self.routing_manager.destinations:
                for y in range(des.y_min, des.y_max + 1):
                    for x in range(des.x_min, des.x_max + 1):
                        img[y][x] = [0, 255, 0]  # green for destinations
            for drp in self.routing_manager.droplets:
                for y in range(drp.y_min, drp.y_max + 1):
                    for x in range(drp.x_min, drp.x_max + 1):
                        img[y][x] = [0, 0, 255]  # blue for droplets
            return img
        else:
            raise RuntimeError('Unknown mode in render')

    def seed(self, seed=None):
        pass

    def close(self):
        """ close render view """
        pass

    def printHealthSatus(self):
        print('### Env Health ###')
        n_bad = np.count_nonzero(self.m_health < 0.2)
        n_mid = np.count_nonzero(self.m_health < 0.5)
        n_ok = np.count_nonzero(self.m_health < 0.8)
        print('Really bad:', n_bad,
              'Halfly degraded:', n_mid - n_bad,
              'Mildly degraded', n_ok - n_mid)

    def addUsage(self):
        # for i in range(self.routing_manager.n_droplets):
        for i, agent in enumerate(self.agents):
            if not self.dones[agent]:
                droplet = self.routing_manager.droplets[i]
                for y in range(droplet.y_min, droplet.y_max + 1):
                    for x in range(droplet.x_min, droplet.x_max + 1):
                        self.m_usage[y][x] += 1

    def updateHealth(self):
        if not self.b_degrade:
            return
        index = self.m_usage > 50.0  # degrade here
        self.m_health[index] = self.m_health[index] * self.m_degrade[index]
        self.m_usage[index] = 0

    def getObs(self):
        observations = {}
        for i, agent in enumerate(self.agents):
            observations[agent] = self.getOneObs(i)
        return observations

    def getOneObs(self, agent_index):
        """
        RGB format of image
        Obstacles - red in layer 0
        Goal      - greed in layer 1
        Droplet   - blue in layer 2
        """
        fov = 15
        obs = np.zeros(shape=(4, fov, fov))
        center_x = self.routing_manager.droplets[agent_index].x_center
        center_y = self.routing_manager.droplets[agent_index].y_center
        origin = (center_x-fov//2, center_y-fov//2)
        # First droplets in 0 layer
        drop = self.routing_manager.droplets[agent_index]
        y_min = 0 if drop.y_min < 0 else drop.y_min
        y_max = self.width - 1 if drop.y_max >= self.width else drop.y_max
        x_min = 0 if drop.x_min < 0 else drop.x_min
        x_max = self.length - 1 if drop.x_max >= self.length else drop.x_max
        for y in range(y_min, y_max + 1):
            for x in range(x_min, x_max + 1):
                n_x, n_y = x-origin[0], y-origin[1]
                if (0 <= n_x < fov) and (0 <= n_y < fov):
                    obs[0][n_y][n_x] = agent_index+1
        # get current droplet's goal layer 1
        des = self.routing_manager.destinations[agent_index]
        y_min = 0 if des.y_min < 0 else des.y_min
        y_max = self.width - 1 if des.y_max >= self.width else des.y_max
        x_min = 0 if des.x_min < 0 else des.x_min
        x_max = self.length - 1 if des.x_max >= self.length else des.x_max
        for y in range(y_min, y_max + 1):
            for x in range(x_min, x_max + 1):
                n_x, n_y = x-origin[0], y-origin[1]
                if (0 <= n_x < fov) and (0 <= n_y < fov):
                    obs[1][n_y][n_x] = agent_index+1
        # other droplets in 2 layer
        for idx, d in enumerate(self.routing_manager.droplets):
            if idx != agent_index:
                y_min = 0 if d.y_min < 0 else d.y_min
                y_max = self.width - 1 if d.y_max >= self.width else d.y_max
                x_min = 0 if d.x_min < 0 else d.x_min
                x_max = self.length - 1 if d.x_max >= self.length else d.x_max
                for y in range(y_min, y_max + 1):
                    for x in range(x_min, x_max + 1):
                        n_x, n_y = x-origin[0], y-origin[1]
                        if (0 <= n_x < fov) and (0 <= n_y < fov):
                            obs[2][n_y][n_x] = idx+1
        # get other's Goal layer 3
        for idx, d in enumerate(self.routing_manager.destinations):
            # if idx != agent_index and (abs(d.x_center-center_x)<=(fov//2+3) or abs(d.y_center-center_y)<=(fov//2+3)):
            if idx != agent_index:
                y_min = 0 if d.y_min < 0 else d.y_min
                y_max = self.width - 1 if d.y_max >= self.width else d.y_max
                x_min = 0 if d.x_min < 0 else d.x_min
                x_max = self.length - 1 if d.x_max >= self.length else d.x_max
                for y in range(y_min, y_max + 1):
                    for x in range(x_min, x_max + 1):
                        n_x = np.clip(x - origin[0], 0, fov-1)
                        n_y = np.clip(y - origin[1], 0, fov-1)
                        obs[3][n_y][n_x] = idx+1
        dir_VEC = np.array([des.x_center-center_x, des.y_center-center_y])
        obs = np.append(obs, dir_VEC)
        return obs

    def get_env_info(self):
        env_info = {"n_actions": len(self.actions),
                    "n_agents": len(self.agents),
                    "obs_shape": self.getOneObs(0).flatten().shape[-1],
                    "episode_limit": self.max_step}
        return env_info
