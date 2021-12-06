from runner import Runner
from common.arguments import get_common_args, get_mixer_args
from dmfb import*

if __name__ == '__main__':
    for i in range(4):
        args = get_common_args()
        args = get_mixer_args(args)
        env = DMFBenv(args.chip_size, args.chip_size, args.drop_num,
                      args.block_num, fov=args.fov, stall=args.stall)
        env_info = env.get_env_info()
        args.n_actions = env_info["n_actions"]
        args.n_agents = env_info["n_agents"]
        args.state_shape = env_info["state_shape"]
        args.obs_shape = env_info["obs_shape"]
        args.episode_limit = env_info["episode_limit"]
        runner = Runner(env, args)
        if not args.evaluate:
            runner.run(i)
        else:
            average_episode_rewards, average_episode_steps, average_episode_constraints, success_rate = runner.evaluate()
            print('The averege total_rewards of {} is  {}'.format(
                args.alg, average_episode_rewards))
            print('The each epoch total_steps is: {}'.format(
                average_episode_steps))
            print('The successful rate is: {}'.format(success_rate))
            break
        env.close()
