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
from meda import*
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
    results = {'multistep': [0]*len_results, 'multi': [0]*len_results,'success':[0]*len_results,
               'baseline': [0]*len_results,'basestep':[0]*len_results}
    health= np.zeros((args.evaluate_epoch,env.width,env.length))
    for i in range(len_results):
        print('### Evaluating iteration %d' %(i))
        model_name = '_'.join(['repeat', '1', 'training', '100', '20000'])
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
        baseline_agent = BaseLineRouter(args.width, args.length)
        health[i] = env.m_health
        for j in range(args.evaluate_episode):
            obs = env.reset()
            routing_manager = env.routing_manager
            results['baseline'][i] += baseline_agent.getEstimatedReward(routing_manager,env.m_health)[0]
            results['basestep'][i] += baseline_agent.getEstimatedReward(routing_manager,env.m_health)[1]
            eposideR,success,step = EvaluateAgent(args, env, obs, multi_agent, args.method == 'centralized')
            results['multi'][i] += eposideR
            results['success'][i]  += success
            results['multistep'][i]+= step
        results['multi'][i] /= args.evaluate_episode
        results['baseline'][i] /= args.n_evaluate
        results['basestep'][i] /= args.n_evaluate
        results['success'][i] /= args.evaluate_episode
        results['multistep'][i] /= args.evaluate_episode
    return results,health

def save_evaluation(agent_rewards, filename, args):
    path = 'DegreData'+'/'+ str(args.n_agents)
    if not os.path.exists(path):
        os.makedirs(path)
    filename = path +'/'+ filename
    np.save(filename,agent_rewards)
def evaluateSeveralTimes(args=None, path_log=None):
    showIsGPU()
    multi_rewards = []
    baseline_rewards = []
    success=[]
    multisteps=[]
    basesteps=[]
    health = []
    for repeat in range(1, args.n_repeat+1):
        print("### In repeat %d" %(repeat))
        start_time = time.time()
        env = MEDAEnv(w=args.width, l=args.length, n_agents=args.n_agents,
                      b_degrade= True, per_degrade = args.per_degrade)
        results,healthy = evaluateOnce(args, path_log, env, repeat_num=repeat)
        print("### Repeat %s costs %s seconds ###" %(str(repeat), time.time() - start_time))
        multi_rewards.append(results['multi'])
        baseline_rewards.append(results['baseline'])
        success.append(results['success'])
        multisteps.append(results['multistep'])
        basesteps.append(results['basestep'])
        health.append(healthy)
    save_evaluation(multi_rewards, 'multi_rewards.npy', args)
    save_evaluation(baseline_rewards, 'baseline_rewards.npy', args)
    save_evaluation(basesteps,'basesteps.npy',args)
    save_evaluation(multisteps,'muti_steps.npy',args)
    save_evaluation(success,'success_rate.npy',args)
    save_evaluation(np.asanyarray(health),'health.npy',args)
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
    parser.add_argument('--n-repeat', help='Number of repeats for the experiment', type=int, default=5)
    parser.add_argument('--n-timesteps', help='Number of timesteps for each iteration',
                        type=int, default=20000)
    # env settings
    parser.add_argument('--width', help='Width of the biochip', type = int, default = 30)
    parser.add_argument('--length', help='Length of the biochip', type = int, default = 60)
    parser.add_argument('--n-agents', help='Number of agents', type = int, default = 2)
    parser.add_argument('--b-degrade', action = "store_true")
    parser.add_argument('--per-degrade', help='Percentage of degrade', type = float, default = 1.0)
    # rl evaluate
    parser.add_argument('--n-evaluate', help='Number of episodes to evaluate the model for each iteration',
                        type=int, default=100)
    parser.add_argument('--evaluate_epoch', type=int, default=20,
                        help='number of the epoch to evaluate the agent')
    parser.add_argument('--evaluate_episode', type=int, default=100,
                        help='number of the epoch to evaluate the agent')
    return parser

def main(args=None):
    np.random.seed(1) 
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
