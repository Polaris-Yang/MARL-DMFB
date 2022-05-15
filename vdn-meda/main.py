from runner import Runner
from common.arguments import get_common_args, get_mixer_args
from meda import *

if __name__ == '__main__':
    for i in range(5):
        args = get_common_args()
        args = get_mixer_args(args)
        print(args)
        env = MEDAEnv(w = args.width,l = args.length,n_agents= args.drop_num,
                      b_degrade= args.b_degrade,per_degrade= args.per_degrade)
        env_info = env.get_env_info()
        args.n_actions = env_info["n_actions"]
        args.n_agents = env_info["n_agents"]
        args.obs_shape = env_info["obs_shape"]
        args.episode_limit = env_info["episode_limit"]
        runner = Runner(env, args)
        if not args.evaluate:
            runner.run(i)
        else:
            average_episode_rewards, average_episode_steps, success_rate = runner.evaluate()
            print('The averege total_rewards of {} is  {}'.format(args.alg, average_episode_rewards))
            print('The each epoch total_steps is: {}'.format(average_episode_steps))
            print('The successful rate is: {}'.format(success_rate))
            break
        env.close()
