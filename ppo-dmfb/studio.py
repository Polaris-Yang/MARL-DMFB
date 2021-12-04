#!/usr/bin/python

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import argparse

import time
from PIL import ImageDraw
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf

from my_net import VggCnnPolicy, VggCnnLstmPolicy, VggCnnLnLstmPolicy

from meda import*

from stable_baselines.common import make_vec_env
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines import PPO2, ACER, DQN
from my_net import VggCnnPolicy, DqnVggCnnPolicy
from utilities import DecentrailizedTrainer, CentralizedEnv, ParaSharingEnv, ConcurrentAgentEnv
import csv


ALGOS = {'PPO': PPO2, 'ACER': ACER, 'DQN': DQN}

def showIsGPU():
    if tf.test.is_gpu_available():
        print("### Training on GPUs... ###")
    else:
        print("### Training on CPUs... ###")

def getGameLength(env):
    ans = env.agt_pos[0] - env.agt_end[0]
    ans += env.agt_pos[1] - env.agt_end[1]
    return ans

def drawGridInImg(image, step):
    draw = ImageDraw.Draw(image)
    y_top = 0
    y_btm = image.height
    for x in range(0, image.width, step):
        line = ((x, y_top), (x, y_btm))
        draw.line(line, fill = "rgb(64, 64, 64)")
    x_lef = 0
    x_rig = image.width
    for y in range(0, image.height, step):
        line = ((x_lef, y), (x_rig, y))
        draw.line(line, fill = "rgb(64, 64, 64)")
    return image

def runAGame(agent, env, centralized = True):
    if centralized:
        env = CentralizedEnv(env)
    obs = env.reset()
    done, state = False, None
    images = []
    while not done:
        if centralized: # centralized multi-agent
            action, state = agent.predict(obs)
            obs, reward, done, _info = env.step(action)
        else: # concurrent, para-sharing
            action = {}
            for droplet in env.agents:
                model = agent[droplet]
                obs_droplet = obs[droplet]
                action[droplet], _ = model.predict(obs_droplet)
            obs, m_rewards, m_dones, _info = env.step(action)
            reward = np.average([r for r in m_rewards.values()])
            done = bool(np.all([d for d in m_dones.values()]))
        img = env.render('rgb_array')
        frame = Image.fromarray(img)
        scale = int(200 / env.width)
        frame = frame.resize((env.length * scale, env.width * scale))
        frame = drawGridInImg(frame, scale)
        images.append(frame)
    return images[:-1]

def recordVideo(args, env, model, filename):
    """ Record videos for three games """
    # env = model.get_env()
    images = []
    images = images + runAGame(model, env, args.method == 'centralized')
    images = images + runAGame(model, env, args.method == 'centralized')
    images = images + runAGame(model, env, args.method == 'centralized')
    images[0].save(filename + '.gif',
            format='GIF',
            append_images=images[1:],
            save_all=True,
            duration=500,
            loop=0)
    print('Video saved:', filename)


def trainNRecordASetting(args, path_log, env, repeat_num=1):
    algo = ALGOS[args.algo]
    len_results = (args.stop_iters - args.start_iters)//5 + 1

    video_name = os.path.join(path_log, 'images')
    Path(video_name).mkdir(parents=True, exist_ok=True)
    # No modules first
    for i in range(len_results):
        print('### Recording iteration %d' %(i*5))
        model_name = '_'.join(['repeat', str(repeat_num), 'training', str(i*5), str(args.n_timesteps)])
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

        recordVideo(args, env, multi_agent, os.path.join(video_name, str(i)))

def recordTrainingProcess(args, path_log):
    showIsGPU()
    env = MEDAEnv(w=args.width, l=args.length, n_agents=args.n_agents,
                  b_degrade=args.b_degrade, per_degrade = args.per_degrade)
    trainNRecordASetting(args, path_log, env)

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
                        type=int, default=100)
    parser.add_argument('--n-timesteps', help='Number of timesteps for each iteration',
                        type=int, default=20000)
    # env settings
    parser.add_argument('--width', help='Width of the biochip', type = int, default = 30)
    parser.add_argument('--length', help='Length of the biochip', type = int, default = 60)
    parser.add_argument('--n-agents', help='Number of agents', type = int, default = 2)
    parser.add_argument('--b-degrade', action = "store_true")
    parser.add_argument('--per-degrade', help='Percentage of degrade', type = float, default = 0.1)
    # rl evaluate
    parser.add_argument('--n-evaluate', help='Number of episodes to evaluate the model for each iteration',
                        type=int, default=20)
    return parser

def main(args=None):
    parser = get_parser()
    args = parser.parse_args(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    # the path to where log files will be saved
    # example path: log/30_60/PPO_SimpleCnnPolicy
    path_log = os.path.join('log', args.method, str(args.width)+'_'+str(args.length),
            str(args.n_agents), args.algo+'_VggCnnPolicy')
    print('### Start recording algorithm %s'%(args.algo))

    recordTrainingProcess(args, path_log = path_log)
    print('### Finished studio.py successfully ###')

if __name__ == '__main__':
    main()
