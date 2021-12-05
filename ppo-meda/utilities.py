#!/usr/bin/python
"""
Trainers that use baseline algorithms for the multi-agent envrionment
2020/12/20 Tung-Che Liang
"""

import os
import gym
import threading
from meda import*
from my_net import VggCnnPolicy, VggCnnLnLstmPolicy, DqnVggCnnPolicy
from stable_baselines import PPO2, ACER, DQN
from stable_baselines.common import make_vec_env

class LearnThread(threading.Thread):
    def __init__(self, model, total_timesteps, agent):
        super(LearnThread, self).__init__()
        self.model = model
        self.total_timesteps = total_timesteps
        self.agent = agent

    def run(self):
        print("### Running thread", self.agent, "...")
        self.model.learn(self.total_timesteps)
        print("### Finished thread", self.agent)

class DecentrailizedTrainer:
    """
    This trainer uses baseline models under the hood for pettingzoo envs
    """
    def __init__(self, policy, parallel_env, model_type, concurrent = True):
        assert(isinstance(parallel_env, ParallelEnv))
        if model_type not in ['PPO', 'ACER', 'DQN']:
            raise TypeError("{} is not a legal model type".format(model_type))
        self.models = {}
        self.p_env = parallel_env
        self.concurrent = concurrent
        if concurrent:
            for agent in parallel_env.agents:
                self.models[agent] = self.getModel(
                        model_type,
                        copy.deepcopy(policy),
                        ConcurrentAgentEnv(parallel_env, agent))
        else: # parameter sharing
            model = self.getModel(
                    model_type,
                    policy,
                    ParaSharingEnv(parallel_env))
            for agent in parallel_env.agents:
                self.models[agent] = model

    def getCurrModels(self):
        return self.models

    def getModel(self, model_type, policy, env):
        if model_type == "PPO":
            return PPO2(policy, env)
        elif model_type == "ACER":
            return ACER(policy, env)
        else: # DQN
            return DQN(policy, env)

    def get_env(self):
        return self.models[self.p_env.agents[0]].get_env()

    def getCurrModels(self):
        return self.models

    def learn(self, total_timesteps):
        if self.concurrent:
            threads = []
            for agent in self.p_env.agents:
                model = self.models[agent]
                thd = LearnThread(model, total_timesteps, agent)
                thd.start()
                threads.append(thd)
            for thd in threads:
                thd.join()
        else:
            self.models[self.p_env.agents[0]].learn(total_timesteps)

    def save(self, save_path):
        if self.concurrent:
            for i, agent in enumerate(self.p_env.agents):
                model = self.models[agent]
                model.save(save_path + '_c{}'.format(i))
        else:
            self.models[self.p_env.agents[0]].save(save_path + 'shared')

    def load(self, save_path, model_type, is_vec=False):
        if self.concurrent:
            for i, agent in enumerate(self.p_env.agents):
                if is_vec:
                    vec_env = make_vec_env(ConcurrentAgentEnv,
                            env_kwargs = {'env': self.p_env, 'agent': agent})
                    self.models[agent] = model_type.load(
                            save_path+'_c{}'.format(i), env=vec_env)
                else:
                    self.models[agent] = model_type.load(
                            save_path+'_c{}'.format(i), env=self.p_env)
        else:
            for i, agent in enumerate(self.p_env.agents):
                if is_vec:
                    vec_env = make_vec_env(ParaSharingEnv,
                            env_kwargs = {'env': self.p_env})
                    self.models[agent] = model_type.load(
                            save_path+'shared', env=vec_env)
                else:
                    self.models[agent] = model_type.load(
                            save_path+'shared', env=self.p_env)

class ConcurrentAgentEnv(gym.Env):
    """ Single Agent for MEDAEnv(ParallelEnv) """
    metaddata = {'render.modes': ['human', 'rgb_array']}
    def __init__(self, env, agent):
        super(ConcurrentAgentEnv, self).__init__()
        self.env = env
        if agent not in self.env.agents:
            raise TypeError("{} is not one of the agents in {}".format(
                    agent, env))
        self.agent = agent
        self.count = 0
        self.agent_index = self.env.agents.index(agent)
        self.action_space = self.env.action_spaces[agent]
        self.observation_space = self.env.observation_spaces[agent]
        self.reward_range = (-1.0, 1.0)

    def step(self, action):
        self.count += 1
        reward = self.env.routing_manager.moveOneDroplet(
                self.agent_index, action, self.env.m_health, True)
        obs = self.env.getOneObs(self.agent_index)
        if self.count <= self.env.max_step:
            done = self.env.routing_manager.getTaskStatus()[self.agent_index]
            self.addMCUsage()
        else:
            done = True
        return obs, reward, done, {}

    def reset(self):
        #print('### Resetting task', self.agent_index, '...')
        self.count = 0
        self.env.routing_manager.resetTask(self.agent_index)
        self.env.updateHealth()
        return self.env.getOneObs(self.agent_index)

    def restart(self):
        return self.env.restart(self.agent_index)

    def render(self, mode = 'human'):
        return self.env.render(mode)

    def close(self):
        pass

    def addMCUsage(self):
        droplet = self.env.routing_manager.droplets[self.agent_index]
        for y in range(droplet.y_min, droplet.y_max + 1):
            for x in range(droplet.x_min, droplet.x_max + 1):
                self.env.m_usage[y][x] += 1

class ParaSharingEnv(gym.Env):
    """ Single Agent for MEDAEnv(ParallelEnv) """
    metaddata = {'render.modes': ['human', 'rgb_array']}
    def __init__(self, env):
        super(ParaSharingEnv, self).__init__()
        self.env = env
        self.agent_index = 0
        self.count = 0 # count loop through all agents
        agent = env.agents[0]
        self.action_space = self.env.action_spaces[agent]
        self.observation_space = self.env.observation_spaces[agent]
        self.reward_range = (-1.0, 1.0)

    def step(self, action):
        reward = self.env.routing_manager.moveOneDroplet(
                self.agent_index, action, self.env.m_health)
        if self.agent_index == len(self.env.agents) - 1:
            self.agent_index = 0
            self.count += 1
        else:
            self.agent_index += 1
        obs = self.env.getOneObs(self.agent_index)
        if self.count > self.env.max_step:
            done = True
        else:
            done = self.env.routing_manager.getTaskStatus()[self.agent_index]
            self.addMCUsage()
        return obs, reward, done, {}

    def reset(self):
        self.env.routing_manager.refresh()
        self.agent_index = 0
        self.count = 0
        self.env.updateHealth()
        return self.env.getOneObs(self.agent_index)

    def restart(self):
        return self.env.restart(self.agent_index)

    def render(self, mode = 'human'):
        return self.env.render(mode)

    def close(self):
        pass

    def addMCUsage(self):
        droplet = self.env.routing_manager.droplets[self.agent_index]
        for y in range(droplet.y_min, droplet.y_max + 1):
            for x in range(droplet.x_min, droplet.x_max + 1):
                self.env.m_usage[y][x] += 1

class CentralizedEnv(gym.Env):
    """ Centralized wrapper for MEDAEnv(ParallelEnv) """
    metaddata = {'render.modes': ['human', 'rgb_array']}
    def __init__(self, env):
        super(CentralizedEnv, self).__init__()
        self.env = env
        n_droplets = len(env.agents)
        self.action_space = spaces.Discrete(len(env.actions) ** n_droplets)
        self.observation_space = spaces.Box(low = 0, high = 1,
                shape = (env.width, env.length, 3 * n_droplets),
                dtype = 'uint8')
        self.reward_range = (-1.0, 1.0)

    def step(self, action):
        m_actions = self.decompressedAction(action)
        m_obs, m_rewards, m_dones, _ = self.env.step(m_actions)
        obs = self.transformObs(m_obs)
        reward = np.average([r for r in m_rewards.values()])
        done = bool(np.all([d for d in m_dones.values()]))
        return obs, reward, done, {}

    def reset(self):
        m_obs = self.env.reset()
        return self.transformObs(m_obs)

    def restart(self):
        m_obs = self.env.restart()
        return self.transformObs(m_obs)

    def render(self, mode = 'human'):
        return self.env.render(mode)

    def close(self):
        pass

    def decompressedAction(self, action):
        l_actions = []
        while action > 0:
            l_actions.append(Action(action % len(Action)))
            action = int(action / len(Action))
        while len(l_actions) < self.env.routing_manager.n_droplets:
            l_actions.append(Action(0))
        return {agent: act for agent, act in zip(self.env.agents, l_actions)}

    def transformObs(self, m_obs):
        return np.concatenate(
                [m_obs[agent] for agent in self.env.agents], axis = 2)

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    p_env = MEDAEnv(30, 60, 4)
    model = DecentrailizedTrainer(VggCnnPolicy, p_env, 'PPO', True)
    model.learn(1000)
    model = DecentrailizedTrainer(DqnVggCnnPolicy, p_env, 'DQN', True)
    model.learn(1000)

    model = DecentrailizedTrainer(VggCnnPolicy, p_env, 'ACER', False)
    model.learn(100)

    # evaluation example
    # 1. Centralized learning
    centralized_env = CentralizedEnv(p_env)
    model = PPO2(VggCnnPolicy, centralized_env)
    if False: # the pseudo code below is not executable
        model.load('path')
        initial_obs = centralized_env.reset()
        done = False
        while done:
            action, state = model.predict(initial_obs)
            obs, reward, done, _ = centralized_env.step(action)

    # restart droplets for concurrent agents
    p_env.restart()
    # 2. Decentralized and concurrent learning
    trainer = DecentrailizedTrainer(VggCnnPolicy, p_env, 'PPO', True)
    if False: # the pseudo code below is not executable
        actions = [0] * len(p_env.agents)
        observations = []
        rewards = [0.0] * len(p_env.agents)
        dones = [False] * len(p_env.agents)
        for i, agent in enumerate(p_env.agents):
            model = trainer.models[agent]
            model.load('path')
            observations.append(initial_obs[:, :, 3*i : 3*i+3])
        while np.all(dones): # when all done
            for i, agent in enumerate(p_env.agents):
                model = trainer.models[agent]
                env = model.envs[0] # might be a vectorized Env
                actions[i], state = model.predict(observations[i])
                observations[i], rewards[i], dones[i], _ = env.step()

    # restart droplets for concurrent agents
    p_env.restart()
    # 3. Decentralized and parameter-sharing learning
    trainer = DecentrailizedTrainer(VggCnnPolicy, p_env, 'PPO', False)
    if False: # the pseudo code below is not executable
        model = trainer.models[p_env.agents[0]]
        model.load('path')
        actions = [0] * len(p_env.agents)
        observations = []
        rewards = [0.0] * len(p_env.agents)
        dones = [False] * len(p_env.agents)
        for i in range(len(p_env.agents)):
            observations.append(initial_obs[:, :, 3*i : 3*i+3])
        while np.all(dones): # when all done
            for i in range(len(p_env.agents)):
                env = model.envs[0] # might be a vectorized Env
                actions[i], state = model.predict(observations[i])
                observations[i], rewards[i], dones[i], _ = env.step()
    print("Finished utilities.py")
