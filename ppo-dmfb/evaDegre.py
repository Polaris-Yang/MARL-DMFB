#!/usr/bin/python
import os
import numpy as np
from matplotlib.pyplot import step
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import argparse
import time
import tensorflow as tf
from stable_baselines.common import make_vec_env
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines import PPO2, ACER, DQN
from dmfb import*
from my_net import VggCnnPolicy, DqnVggCnnPolicy
from utilities import DecentrailizedTrainer, ConcurrentAgentEnv
import csv

ALGOS = {'PPO': PPO2, 'ACER': ACER, 'DQN': DQN}

def showIsGPU():
    if tf.test.is_gpu_available():
        print("### Training on GPUs... ###")
    else:
        print("### Training on CPUs... ###")

def EvaluateAgent(args, env, obs, agent, centralized = True):
    obs = env.restart()
    episode_reward = 0.0
    done, state = False, None
    step = 0
    success = 0.0
    while not done:
        action = {}
        for droplet in env.agents:
            model = agent[droplet]
            obs_droplet = obs[droplet]
            action[droplet], _ = model.predict(obs_droplet)
        obs, m_rewards, m_dones, _info = env.step(action) # env: Meda, obs: m_obs
        reward = np.average([r for r in m_rewards.values()])
        done = bool(np.all([d for d in m_dones.values()]))
        step+=1
        episode_reward += reward
        success += _info['success']
    return episode_reward,success,step

def evaluateOnce(args, path_log, env, repeat_num):
    algo = ALGOS[args.algo]
    len_results = args.evaluate_epoch
    results = {'multistep': [0]*len_results, 'multi': [0]*len_results,'success':[0]*len_results}
    for i in range(len_results):
        print('### Evaluating iteration %d' %(i))
        model_name = '_'.join(['repeat', '1', 'training', '250', '20000'])
        path_multi = os.path.join(path_log, model_name)
        if args.method == 'centralized':
            multi_agent = algo.load(path_multi)
        else:
            multi_agent = {}
            for agent_index, agent in enumerate(env.agents):
                if args.method == 'concurrent':
                    multi_agent[agent] = algo.load(path_multi+'_c{}'.format(agent_index))
                else:
                    multi_agent[agent] = algo.load(path_multi+'shared')
        for j in range(args.evaluate_episode):
            obs = env.reset()
            routing_manager = env.routing_manager
            eposideR,success,step = EvaluateAgent(args, env, obs, multi_agent, args.method == 'centralized')
            results['multi'][i] += eposideR
            results['success'][i]  += success
            results['multistep'][i]+= step
        results['multi'][i] /= args.evaluate_episode
        results['success'][i] /= args.evaluate_episode
        results['multistep'][i] /= args.evaluate_episode
    return results

def save_evaluation(agent_rewards, filename, path_log):
    # with open(os.path.join(path_log, filename), 'w') as agent_log:
    #     writer_agent = csv.writer(agent_log)
    #     writer_agent.writerows(agent_rewards)filename
    np.save(filename,agent_rewards)

def evaluateSeveralTimes(args=None, path_log=None):
    showIsGPU()
    multi_rewards = []
    success=[]
    multisteps=[]
    for repeat in range(1, args.n_repeat+1):
        print("### In repeat %d" %(repeat))
        start_time = time.time()
        env = DMFBenv(width=args.width, length=args.length, n_agents=args.n_agents,n_blocks=0,
                      b_degrade=True, per_degrade = 0.5)
        results = evaluateOnce(args, path_log, env, repeat_num=repeat)
        print("### Repeat %s costs %s seconds ###" %(str(repeat), time.time() - start_time))
        multi_rewards.append(results['multi'])
        success.append(results['success'])
        multisteps.append(results['multistep'])
    save_evaluation(multi_rewards, 'multi_rewards.npy', path_log)
    save_evaluation(multisteps,'muti_steps.npy',path_log)
    save_evaluation(success,'success_rate.npy',path_log)
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
    parser.add_argument('--n-repeat', help='Number of repeats for the experiment', type=int, default=1)
    parser.add_argument('--n-timesteps', help='Number of timesteps for each iteration',
                        type=int, default=20000)
    # env settings
    parser.add_argument('--width', help='Width of the biochip', type = int, default = 10)
    parser.add_argument('--length', help='Length of the biochip', type = int, default = 10)
    parser.add_argument('--n-agents', help='Number of agents', type = int, default = 2)
    parser.add_argument('--b-degrade', action = "store_true")
    parser.add_argument('--per-degrade', help='Percentage of degrade', type = float, default = 0.5)
    # rl evaluate
    parser.add_argument('--n-evaluate', help='Number of episodes to evaluate the model for each iteration',
                        type=int, default=100)
    parser.add_argument('--evaluate_epoch', type=int, default=30,
                        help='number of the epoch to evaluate the agent')
    parser.add_argument('--evaluate_episode', type=int, default=500,
                        help='number of the epoch to evaluate the agent')
    return parser

def main(args=None):
    parser = get_parser()
    args = parser.parse_args(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    # the path to where log files will be saved
    # example path: log/30_60/PPO_SimpleCnnPolicy
    path_log = os.path.join('log', args.method, str(args.width)+'_'+str(args.length),
            str(args.n_agents), args.algo+'_VggCnnPolicy')
    print('### Start evaluating algorithm %s'%(args.algo))
    evaluateSeveralTimes(args, path_log = path_log)

if __name__ == '__main__':
    main()
