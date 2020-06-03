import gym

import argparse

import collections
import os
import sys
import json
import datetime

from wrapper import NumpyFloat32Env, ScreenRenderEnv

from tensorboardX import SummaryWriter

from sac import SAC as OriginalSAC
from sac_extended import SAC as ExtendedSAC
from models.actors import MujocoActor
from models.critics import QFunction, VFunction


def build_env(args):
    env = gym.make(args.env)
    env = NumpyFloat32Env(env)
    if args.render:
        env = ScreenRenderEnv(env)
    return env


def write_text_to_file(file_path, data):
    with open(file_path, 'w') as f:
        f.write(data)


def create_dir_if_not_exist(outdir):
    if os.path.exists(outdir):
        if not os.path.isdir(outdir):
            raise RuntimeError('{} is not a directory'.format(outdir))
        else:
            return
    os.makedirs(outdir)


def prepare_output_dir(base_dir, args, time_format='%Y-%m-%d-%H%M%S'):
    time_str = datetime.datetime.now().strftime(time_format)
    outdir = os.path.join(base_dir, time_str)
    create_dir_if_not_exist(outdir)

    # Save all the arguments
    args_file_path = os.path.join(outdir, 'args.txt')
    if isinstance(args, argparse.Namespace):
        args = vars(args)
    write_text_to_file(args_file_path, json.dumps(args))

    # Save the command
    argv_file_path = os.path.join(outdir, 'command.txt')
    argv = ' '.join(sys.argv)
    write_text_to_file(argv_file_path, argv)

    return outdir


def prepare_summary_dir(base_dir, time_format='%Y-%m-%d-%H%M%S'):
    time_str = datetime.datetime.now().strftime(time_format)
    summarydir = os.path.join(base_dir, time_str)
    create_dir_if_not_exist(summarydir)

    return summarydir


def load_params(sac, args):
    print('loading model params')
    if args.extended:
        sac.load_models(args.q1_params, args.q2_params,
                        args.pi_params, args.alpha_params)
    else:
        sac.load_models(args.v_params, args.q1_params,
                        args.q2_params, args.pi_params)


def save_params(sac, timestep, outdir, args):
    print('saving model params of iter: ', timestep)
    sac.save_models(outdir, timestep)


def run_training_loop(train_env, eval_env, sac, args):
    algorithm_name = 'ExtendedSAC' if args.extended else 'OriginalSAC'
    base_dir = args.outdir + '/' + algorithm_name
    outdir = prepare_output_dir(base_dir=base_dir, args=args)
    summarydir = prepare_summary_dir(base_dir=args.outdir)

    writer = SummaryWriter(logdir=summarydir)

    for timestep in range(1, args.total_timesteps + 1, 1):
        sac.train(train_env)
        if timestep % args.evaluation_interval == 0:
            _, mean, median, std = sac.evaluate(eval_env)
            print(
                "evaluation at step: {}. mean: {} +/- {}, median: {}".format(timestep, mean, std, median))
            sac.save_models(outdir, timestep)
            writer.add_scalars(
                'eval_result', {'mean': mean, 'median': median}, global_step=timestep)


def pi_builder(state_dim, action_dim):
    return MujocoActor(state_dim=state_dim, action_dim=action_dim)


def q_func_builder(state_dim, action_dim):
    return QFunction(state_dim=state_dim, action_dim=action_dim)


def v_func_builder(state_dim):
    return VFunction(state_dim=state_dim)


def start_training(args):
    train_env = build_env(args)
    eval_env = build_env(args)
    if args.extended:
        sac = ExtendedSAC(
            q_func_builder=q_func_builder,
            pi_builder=pi_builder,
            state_dim=train_env.observation_space.shape[0],
            action_dim=train_env.action_space.shape[0],
            lr=args.learning_rate,
            gamma=args.gamma,
            tau=args.tau,
            batch_size=args.batch_size,
            environment_steps=args.environment_steps,
            gradient_steps=args.gradient_steps,
            device=args.gpu)
    else:
        sac = OriginalSAC(
            v_func_builder=v_func_builder,
            q_func_builder=q_func_builder,
            pi_builder=pi_builder,
            state_dim=train_env.observation_space.shape[0],
            action_dim=train_env.action_space.shape[0],
            lr=args.learning_rate,
            gamma=args.gamma,
            tau=args.tau,
            batch_size=args.batch_size,
            environment_steps=args.environment_steps,
            gradient_steps=args.gradient_steps,
            device=args.gpu)
    load_params(sac, args)

    run_training_loop(train_env, eval_env, sac, args)

    train_env.close()
    eval_env.close()


def main():
    parser = argparse.ArgumentParser()

    # output
    parser.add_argument('--outdir', type=str, default='results')

    # Environment
    parser.add_argument('--env', type=str, default='Walker2d-v2')
    parser.add_argument('--render', action="store_true")

    # Gpu
    parser.add_argument('--gpu', type=int, default=-1)

    # testing
    # parser.add_argument('--test-run', action='store_true')
    # parser.add_argument('--save-video', action='store_true')

    # params
    parser.add_argument('--v-params', type=str, default="")
    parser.add_argument('--q1-params', type=str, default="")
    parser.add_argument('--q2-params', type=str, default="")
    parser.add_argument('--pi-params', type=str, default="")
    parser.add_argument('--alpha-params', type=str, default="")
    parser.add_argument('--extended', action="store_true")

    # Training parameters
    parser.add_argument('--total-timesteps', type=int, default=1000000)
    parser.add_argument('--learning-rate', type=float, default=3.0*1e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--start-timesteps', type=int, default=1000)
    parser.add_argument('--environment-steps', type=int, default=1)
    parser.add_argument('--gradient-steps', type=int, default=1)
    parser.add_argument('--evaluation-interval', type=float, default=5000)

    args = parser.parse_args()

    start_training(args)


if __name__ == "__main__":
    main()
