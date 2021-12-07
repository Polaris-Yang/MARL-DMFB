#!/usr/bin/python
"""
Trainers that use baseline algorithms for the multi-agent envrionment
2020/12/20 Tung-Che Liang
"""

import os
import gym
import threading
from dmfb import*
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
        # print("### Running thread", self.agent, "...")
        self.model.learn(self.total_timesteps)
        # print("### Finished thread", self.agent)

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
    def getCurrModels(self):
        return self.models

    def getModel(self, model_type, policy, env):
        if model_type == "PPO":
            return PPO2(policy, env)
        elif model_type == "ACER":
            return ACER(policy, env)

    def get_env(self):
        return self.models[self.p_env.agents[0]].get_env()

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

    def save(self, save_path):
        if self.concurrent:
            for i, agent in enumerate(self.p_env.agents):
                model = self.models[agent]
                model.save(save_path + '_c{}'.format(i))

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
        self.reward_range = (-30.0, 30.0)

    def step(self, action):
        self.count += 1
        reward,_,_ = self.env.routing_manager.moveOneDroplet(
                self.agent_index, action, self.env.m_health, True)
        reward -= 2*self.comflic_static()
        reward -= 2*self.comflic_dynamic()
        if np.all(self.env.routing_manager.getTaskStatus()):
            reward+=20
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
        self.env.m_usage[droplet.y][droplet.x] += 1

    def comflic_static(self):
        static_conflic = 0
        cur_positions = self.env.routing_manager.curs
        for i in range(len(self.env.agents)):
            if i==self.agent_index:
                continue
            if np.linalg.norm(cur_positions[i]-cur_positions[self.agent_index]) < 2:
                static_conflic += 1
        return static_conflic

    def comflic_dynamic(self):
        dynamic_conflict = 0
        cur_position = self.env.routing_manager.curs
        past_pisition = self.env.routing_manager.pasts
        for i in range(len(self.env.agents)):
            if i==self.agent_index:
                continue
            if np.linalg.norm(past_pisition[self.agent_index] - cur_position[i]) < 2:
                dynamic_conflict += 1
        for i in range(len(self.env.agents)):
            if i==self.agent_index:
                continue
            if np.linalg.norm(past_pisition[i] - cur_position[self.agent_index]) < 2:
                dynamic_conflict += 1
        return dynamic_conflict
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    p_env = DMFBenv(10, 10, 2)

    # 1. Decentralized and concurrent learning
    trainer = DecentrailizedTrainer(VggCnnPolicy, p_env, 'PPO', True)
    trainer.learn(1000)
    print(trainer)
    print(trainer.models)
    print(trainer.p_env)
