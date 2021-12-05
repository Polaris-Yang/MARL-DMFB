#!/usr/bin/python
import os

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
from utilities import DecentrailizedTrainer, CentralizedEnv, ParaSharingEnv, ConcurrentAgentEnv
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
    len_results = (args.stop_iters - args.start_iters)//2 + 1
    results = {'baseline': [0]*len_results, 'multistep': [0]*len_results, 'multi': [0]*len_results,'success':[0]*len_results,'basestep':[0]*len_results}
    for i in range(len_results):
        print('### Evaluating iteration %d' %(i*2))
        model_name = '_'.join(['repeat', str(repeat_num), 'training', str(i*2), str(args.n_timesteps)])
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
        for j in range(args.n_evaluate):
            if j%5 == 0:
                print('### Episode %d.'%j)
            obs = env.reset()
            routing_manager = env.routing_manager
            results['baseline'][i] += baseline_agent.getEstimatedReward(routing_manager)[0]
            results['basestep'][i] += baseline_agent.getEstimatedReward(routing_manager)[1]
            eposideR,success,step = EvaluateAgent(args, env, obs, multi_agent, args.method == 'centralized')
            results['multi'][i] += eposideR
            results['success'][i]  += success
            results['multistep'][i]+= step
        results['baseline'][i] /= args.n_evaluate
        results['multi'][i] /= args.n_evaluate
        results['success'][i] /= args.n_evaluate
        results['multistep'][i] /= args.n_evaluate
        results['basestep'][i] /= args.n_evaluate
    return results

def save_evaluation(agent_rewards, filename, path_log):
    # with open(os.path.join(path_log, filename), 'w') as agent_log:
    #     writer_agent = csv.writer(agent_log)
    #     writer_agent.writerows(agent_rewards)
    filepath = path_log+ "/" + filename
    np.save(filepath,agent_rewards)

def evaluateSeveralTimes(args=None, path_log=None):
    showIsGPU()
    multi_rewards = []
    baseline_rewards = []
    success=[]
    multisteps=[]
    basesteps=[]
    for repeat in range(1, args.n_repeat+1):
        print("### In repeat %d" %(repeat))
        start_time = time.time()
        env = MEDAEnv(w=args.width, l=args.length, n_agents=args.n_agents,
                      b_degrade=args.b_degrade, per_degrade = args.per_degrade)
        results = evaluateOnce(args, path_log, env, repeat_num=repeat)
        print("### Repeat %s costs %s seconds ###" %(str(repeat), time.time() - start_time))
        multi_rewards.append(results['multi'])
        baseline_rewards.append(results['baseline'])
        success.append(results['success'])
        multisteps.append(results['multistep'])
        basesteps.append(results['basestep'])
    save_evaluation(multi_rewards, 'multi_rewards.npy', path_log)
    save_evaluation(baseline_rewards, 'baseline_rewards.npy', path_log)
    save_evaluation(basesteps,'basesteps.npy',path_log)
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
    # rl evaluate
    parser.add_argument('--n-evaluate', help='Number of episodes to evaluate the model for each iteration',
                        type=int, default=100)
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
