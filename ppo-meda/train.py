#!/usr/bin/python
import os
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
from utilities import DecentrailizedTrainer, CentralizedEnv, ParaSharingEnv, ConcurrentAgentEnv

ALGOS = {'PPO': PPO2, 'ACER': ACER, 'DQN': DQN}

def showIsGPU():
    if tf.test.is_gpu_available():
        print("### Training on GPUs... ###")
    else:
        print("### Training on CPUs... ###")

def runAnExperiment(args, path_log, env, repeat_num):

    algo = ALGOS[args.algo]
    model_name = '_'.join(['repeat', str(repeat_num), 'training', str(args.start_iters), str(args.n_timesteps)])

    if args.algo == 'DQN':
        policy = DqnVggCnnPolicy
    else:
        policy = VggCnnPolicy

    if args.method == 'centralized':
        env = CentralizedEnv(env)
        if args.start_iters == 0:
            model = algo(policy, env)
            if path_log:
                model.save(os.path.join(path_log, model_name))
        else:
            env = make_vec_env(CentralizedEnv, env_kwargs = {'env': env.env})
            model = algo.load(os.path.join(path_log, model_name), env=env)
    else:
        model = DecentrailizedTrainer(policy, env, args.algo, args.method == 'concurrent')
        if args.start_iters == 0:
            if path_log:
                model.save(os.path.join(path_log, model_name))
        else:
            model.load(os.path.join(path_log, model_name), algo, is_vec=True)

    for i in range(args.start_iters+1, args.stop_iters + 1):
        print("### Start training the iteration %d" %(i))
        model.learn(total_timesteps = args.n_timesteps)
        if path_log and i%2 == 0:
            model_name = '_'.join(['repeat', str(repeat_num), 'training', str(i), str(args.n_timesteps)])
            model.save(os.path.join(path_log, model_name))

def expSeveralRuns(args=None, path_log=None):

    showIsGPU()

    for repeat in range(3, 5):
        print("### In repeat %d" %(repeat))
        start_time = time.time()
        env = MEDAEnv(w=args.width, l=args.length, n_agents=args.n_agents,
                      b_degrade=args.b_degrade, per_degrade = args.per_degrade)
        runAnExperiment(args, path_log, env, repeat_num=repeat)
        print("### Repeat %s costs %s seconds ###" %(str(repeat), time.time() - start_time))

def get_parser():
    """
    Creates an argument parser.
    """
    parser = argparse.ArgumentParser(description='RL training for MEDA')

    # device
    parser.add_argument('--cuda', help='CUDA Visible devices', default='0', type=str, required=False)

    # rl algorithm
    parser.add_argument('--algo', help='RL Algorithm', default='PPO', type=str, required=False, choices=list(ALGOS.keys()))
    # rl training
    parser.add_argument('--method', help='The method use for rl training (centralized, sharing, concurrent)',
                        type=str, default='concurrent', choices=['centralized', 'sharing', 'concurrent'])
    parser.add_argument('--n-repeat', help='Number of repeats for the experiment', type=int, default=4)
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
    parser.add_argument('--b-degrade', action = "store_true")
    parser.add_argument('--per-degrade', help='Percentage of degrade', type = float, default = 0)

    return parser

def main(args=None):
    parser = get_parser()
    args = parser.parse_args(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

    # the path to where log files will be saved
    # example path: log/30_60/PPO_SimpleCnnPolicy
    path_log = os.path.join('log', args.method, str(args.width)+'_'+str(args.length),
            str(args.n_agents), args.algo+'_VggCnnPolicy')
    Path(path_log).mkdir(parents=True, exist_ok=True) # create the path if it does not exist

    print('### Start training algorithm %s ###' %(args.algo))
    print('### Log files will be saved to %s ###' %(path_log))

    start_time = time.time()
    expSeveralRuns(args, path_log = path_log)

    print('### Finished algorithm %s successfully ###' %(args.algo))
    print('### Training algorithm %s costs %s seconds ###' %(args.algo, time.time() - start_time))

if __name__ == '__main__':
    main()
