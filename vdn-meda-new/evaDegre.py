from numpy.core.fromnumeric import size
from meda import*
from agent.agent import Agents
import argparse
from common.arguments import get_mixer_args
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import numpy as np
import os
def getparameter():
    parser = argparse.ArgumentParser()
    # the environment setting
    parser.add_argument('--seed', type=int, default=12, help='random seed')
    parser.add_argument('--replay_dir', type=str, default='',
                        help='absolute path to save the replay')
    parser.add_argument('--alg', type=str, default='vdn',
                        help='the algorithm to train the agent')
    parser.add_argument('--n_steps', type=int,
                        default=2000000, help='total time steps')
    parser.add_argument('--n_episodes', type=int, default=10,
                        help='the number of episodes before once training')
    parser.add_argument('--last_action', default=True, action='store_false',
                        help='whether to use the last action to choose action')
    parser.add_argument('--reuse_network', default=True, action='store_false',
                        help='whether to use one network for all agents')
    parser.add_argument('--gamma', type=float,
                        default=0.99, help='discount factor')
    parser.add_argument('--cuda', default=True, action='store_false',
                        help='whether to use the GPU')
    parser.add_argument('--optimizer', type=str,
                        default="ADAM", help='optimizer')
    parser.add_argument('--evaluate_cycle', type=int,
                        default=100000, help='how often to evaluate the model')
    parser.add_argument('--model_dir', type=str,
                        default='./model', help='model directory of the policy')
    parser.add_argument('--result_dir', type=str,
                        default='./result', help='result directory of the policy')
    parser.add_argument('--load_model', default=True, action='store_true',
                        help='whether to load the pretrained model')
    parser.add_argument('--evaluate', default=False, action='store_true',
                        help='whether to evaluate the model')
    parser.add_argument('--evaluate_epoch', type=int, default=10,
                        help='number of the epoch to evaluate the agent')
    parser.add_argument('--evaluate_episode', type=int, default=10,
                        help='number of the epoch to evaluate the agent')
    parser.add_argument('--rnn_hidden_dim', type=int,default=128, help='rnn')
    parser.add_argument('--width', help='Width of the biochip', type = int, default = 30)
    parser.add_argument('--length', help='Length of the biochip', type = int, default = 60)
    parser.add_argument('--drop_num', type=int, default=2, help='the number of droplet')
    parser.add_argument('--b-degrade', action = "store_true")
    parser.add_argument('--per-degrade', help='Percentage of degrade', type = float, default = 1)
    args = parser.parse_args()
    return args

class evaluator():
    def __init__(self, env, args):
        self.agents = Agents(args)
        self.args = args
        self.env = env

    def generate_episode(self):
        self.env.reset()
        terminated = False
        step = 0
        success = 0
        reward = 0  # cumulative rewards
        last_action = np.zeros((self.args.n_agents, self.args.n_actions))
        self.agents.policy.init_hidden(1)
        # epsilon
        epsilon = 0
        while not terminated and step < self.args.episode_limit:
            obs = self.env.getObs()
            obs = [obs[agent].reshape(-1) for agent in self.env.agents]
            actions, avail_actions, actions_onehot = [], [], []
            for agent_id in range(self.args.n_agents):
                avail_action = [1] * 8
                action = self.agents.choose_action(
                    obs[agent_id], last_action[agent_id], agent_id, avail_action, epsilon)
                # generate onehot vector of th action
                action_onehot = np.zeros(self.args.n_actions)
                action_onehot[action] = 1
                actions.append(np.int(action))
                actions_onehot.append(action_onehot)
                avail_actions.append(avail_action)
                last_action[agent_id] = action_onehot
            _, r, terminated, info = self.env.step(actions)
            success += info['success']
            r = np.sum([r[agent] for agent in self.env.agents]) / len(r)
            terminated = np.all([terminated[agent]
                                for agent in self.env.agents])
            reward += r
            step += 1
        return reward, step, success

    def evaluateOneEpoch(self):
        epoch_rewards = 0
        epoch_steps = 0
        epoch_suceess = 0
        for i in range(args.evaluate_episode):
            a, b, c = self.generate_episode()
            epoch_rewards += a
            epoch_steps += b
            epoch_suceess += c
        epoch_rewards = epoch_rewards/args.evaluate_episode
        epoch_step = epoch_steps/args.evaluate_episode
        epoch_suceess = epoch_suceess/args.evaluate_episode
        return epoch_rewards, epoch_step, epoch_suceess

    def evaluate(self):
        epoch_rewards = []
        epoch_steps = []
        epoch_success = []
        # health= np.zeros((args.evaluate_epoch,env.width,env.length))
        for epoch in trange(args.evaluate_epoch):
            # print(self.env.routing_manager.m_health)
            # health[epoch]=self.env.routing_manager.m_health
            rewards, steps, success = self.evaluateOneEpoch()
            epoch_rewards.append(rewards)
            epoch_steps.append(steps)
            epoch_success.append(success)
        return epoch_rewards, epoch_steps, epoch_success


if __name__ == '__main__':
    args = getparameter()
    args = get_mixer_args(args)
    t_rewads=[]
    t_steps=[]
    t_succe=[]
    # health = []
    np.random.seed(1) 
    for i in range(5):    
        env=MEDAEnv(w = args.width,l = args.length,n_agents= args.drop_num,
                b_degrade= True,per_degrade= 1.0)
        env_info = env.get_env_info()
        args.n_actions = env_info["n_actions"]
        args.n_agents = env_info["n_agents"]
        args.obs_shape = env_info["obs_shape"]
        args.episode_limit = env_info["episode_limit"]
        Eva = evaluator(env,args)
        rewards,steps,success=Eva.evaluate()
        t_steps.append(steps)
        t_succe.append(success)
        t_rewads.append(rewards)
        # health.append(healthy)
    path = 'DegreData' + '/' + str(args.drop_num)+'/'
    if not os.path.exists(path):
        os.makedirs(path)
    # health = np.asanyarray(health)
    np.save(path+'rewards.npy',t_rewads)
    np.save(path+'steps.npy',t_steps)
    np.save(path+'success.npy',t_succe)
    # np.save(path+'health.npy',health)


