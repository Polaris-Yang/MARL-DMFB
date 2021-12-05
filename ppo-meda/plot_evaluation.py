#!/usr/bin/python

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import argparse
import time

import matplotlib
import matplotlib.pyplot as plt
# import tensorflow as tf

# from stable_baselines.common import make_vec_env
# from stable_baselines.common.vec_env import DummyVecEnv
# from stable_baselines.common.evaluation import evaluate_policy
# from stable_baselines import PPO2, ACER, DQN

from meda import*
# from my_net import VggCnnPolicy, DqnVggCnnPolicy
import csv

ALGOS = {'PPO': 'PPO2', 'ACER': 'ACER', 'DQN': 'DQN'}

def get_ave_max_min (agent_rewards):
    agent_line = np.average(agent_rewards, axis = 0)
    agent_max = np.max(agent_rewards, axis = 0)
    agent_min = np.min(agent_rewards, axis = 0)
    return agent_line, agent_max, agent_min

def plotAgentPerformance(args, multi_agent_rewards, baseline_rewards, path_log):
    multi_agent_ave, multi_agent_max, multi_agent_min = get_ave_max_min (multi_agent_rewards)
    baseline_ave, baseline_max, baseline_min = get_ave_max_min (baseline_rewards)
    episodes = list(range(len(multi_agent_ave)))
    print(baseline_ave)
    print(multi_agent_ave)
    with plt.style.context('ggplot'):
        plt.rcParams.update({'font.size': 20})
        plt.figure()
        plt.fill_between(episodes, multi_agent_max, multi_agent_min, facecolor = 'red', alpha = 0.3)
        plt.fill_between(episodes, baseline_max, baseline_min, facecolor = 'blue', alpha = 0.3)
        plt.plot(episodes, multi_agent_ave, 'r-', label = 'Multi Agent')
        plt.plot(episodes, baseline_ave, 'b-', label = 'Baseline')
        leg = plt.legend(loc = 'lower right', shadow = True, fancybox = True)
        leg.get_frame().set_alpha(0.5)
        # fig_title = args.algo
        # plt.title(fig_title)
        plt.xlabel('Training Epochs')
        plt.ylabel('Score')
        plt.tight_layout()
        if args.b_degrade:
            path_fig = os.path.join(path_log, 'plot_evluation_degrade.png')
        else:
            path_fig = os.path.join(path_log, 'plot_evluation.png')
        plt.savefig(path_fig)

def read_rewards(path_log, filename):
    path_reward = os.path.join(path_log, filename)
    if not os.path.exists(path_reward):
        raise Exception('Path %s does not exist' %path_reward)
    with open(path_reward, "r") as a:
        wr_a = csv.reader(a, delimiter=',')
        a_rewards=[]
        for row in wr_a:
            if row!=[]:
                a_rewards.append(list(row))
    a_rewards = np.array(a_rewards).astype(np.float)
    return a_rewards

def get_parser():
    """
    Creates an argument parser.
    """
    parser = argparse.ArgumentParser(description='RL training for MEDA')
    # device
    parser.add_argument('--cuda', help='CUDA Visible devices', default='0', type=str, required=False)
    parser.add_argument('--algo', help='RL Algorithm', default='PPO', type=str, required=False, choices=list(ALGOS.keys()))
    # rl training
    parser.add_argument('--method', help='The method use for rl training (centralized, sharing, concurrent)',
                        type=str, default='concurrent', choices=['centralized', 'sharing', 'concurrent'])
    parser.add_argument('--n-repeat', help='Number of repeats for the experiment', type=int, default=3)
    parser.add_argument('--start-iters', help='Number of iterations the initialized model has been trained',
                        type=int, default=0)
    parser.add_argument('--stop-iters', help='Total number of iterations (including pre-train) for one repeat of the experiment',
                        type=int, default=150)
    parser.add_argument('--n-timesteps', help='Number of timesteps for each iteration',
                        type=int, default=20000)
    # env settings
    parser.add_argument('--width', help='Width of the biochip', type = int, default = 30)
    parser.add_argument('--length', help='Length of the biochip', type = int, default = 60)
    parser.add_argument('--n-agents', help='Number of agents', type = int, default = 4)
    parser.add_argument('--b-degrade',default=False,action = "store_true")
    parser.add_argument('--per-degrade', help='Percentage of degrade', type = float, default = 0.1)
    # rl evaluate
    parser.add_argument('--n-evaluate', help='Number of episodes to evaluate the model for each iteration',
                        type=int, default=200)
    return parser

def main(args=None):
    parser = get_parser()
    args = parser.parse_args(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    # the path to where log files will be saved
    # example path: log/30_60/PPO_SimpleCnnPolicy
    path_log = os.path.join('log', args.method, str(args.width)+'_'+str(args.length),
            str(args.n_agents), args.algo+'_VggCnnPolicy')
    print('### Start plotting algorithm %s'%(args.algo))
    if args.b_degrade:
        multi_rewards = read_rewards(path_log, 'multi_degrade_rewards.csv')
        baseline_rewards = read_rewards(path_log, 'baseline_degrade_rewards.csv')
    else:
        multi_rewards = read_rewards(path_log, 'multi_rewards.csv')
        baseline_rewards = read_rewards(path_log, 'baseline_rewards.csv')
    plotAgentPerformance(args, multi_rewards, baseline_rewards, path_log)

if __name__ == '__main__':
    main()
