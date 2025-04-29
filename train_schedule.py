#!/usr/bin/env python
import sys
import os
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path

import torch

from onpolicy.config import get_config

# from onpolicy.envs.gridworld.GridWorld_Env import GridWorldEnv
from onpolicy.envs.rul_schedule.schedule import async_scheduling
from onpolicy.envs.env_wrappers import ScheduleEnv

def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            env = async_scheduling(all_args)
            return env
        return init_env

    return ScheduleEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            env = async_scheduling(all_args)
            return env
        return init_env

    return ScheduleEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument('--scenario_name', type=str, default='rul_schedule', help="Which scenario to run on")
    parser.add_argument('--num_agents', type=int, default=12, help="number of trucks")
    parser.add_argument('--num_factory', type=int, default=50, help="number of factories")
    parser.add_argument('--max_steps', type=int, default=800, help="Max step of each episode and max env step")
    parser.add_argument("--use_single_reward", action='store_true', default=False,
                        help="use single reward")
    parser.add_argument("--use_complete_reward", action='store_true', default=True,
                        help="use complete reward")            
    parser.add_argument("--use_merge", action='store_true', default=False,
                        help="use merge information")
    parser.add_argument("--use_merge_plan", action='store_true', default=False,
                        help="use merge information")
    parser.add_argument("--use_constrict_map", action='store_true', default=False,
                        help="use merge information")
    parser.add_argument("--use_multiroom", action='store_true', default=False,
                        help="use multiroom")
    parser.add_argument("--use_irregular_room", action='store_true', default=False,
                        help="use irregular room")
    parser.add_argument("--use_random_pos", action='store_true', default=False,
                        help="use complete reward")   
    parser.add_argument("--use_time_penalty", action='store_true', default=False,
                        help="use time penalty")    
    parser.add_argument("--use_overlap_penalty", action='store_true', default=False,
                        help="use time penalty")
    parser.add_argument("--use_intrinsic_reward", action='store_true', default=False,
                        help="use intrinsic reward")   
    parser.add_argument("--use_fc_net", action='store_true', default=False,
                        help="use mlp net")  
    parser.add_argument("--use_agent_id", action='store_true', default=False,
                        help="use mlp net")  
    parser.add_argument("--use_global_goal", action='store_true', default=False,
                        help="use global map to choose goal")   
    parser.add_argument("--use_orientation", action='store_true', default=False,
                        help="use agents' orientation info")         
    parser.add_argument("--visualize_input", action='store_true', default=False,
                        help="by default, do not render the env during training. If set, start render. Note: something, the environment has internal render process which is not controlled by this hyperparam.")
    parser.add_argument("--use_up_agents", action='store_true', default=False,
                        help="use_up_agents")         
    parser.add_argument("--use_down_agents", action='store_true', default=False,
                        help="by default.")
    parser.add_argument('--up_agents_step', type=int, default=100, help="local_goal_step")
    parser.add_argument('--down_agents_step', type=int, default=100, help="local_goal_step")
    parser.add_argument('--use_discrect', default = False, action='store_true')
    parser.add_argument('--use_agent_obstacle',action='store_true', default=False)
    # eval by time step
    parser.add_argument('--use_time', default=False, action='store_true')
    # parser.add_argument('--max_timestep', default=200., type=float)
    parser.add_argument('--grid_goal', default = False, action='store_true')
    parser.add_argument('--goal_grid_size', default=4, type=int)
    parser.add_argument('--cnn_trans_layer', type=str, default='1,3,1,1')
    parser.add_argument('--attn_depth', type=int, default=2, help="""attn_depth""")
    parser.add_argument('--use_stack', default = False, action='store_true')
    parser.add_argument('--astar_cost_mode', default = 'normal', choices = ['normal', 'utility'])
    parser.add_argument('--asynch', default=True, action='store_true', help="asynchronized execution")

    # RUL prediction
    parser.add_argument('--use_rul_agent', default = True, action='store_true', help="Use agent to predict RUL")
    parser.add_argument('--rul_threshold', default = 7, type=float, help="RUL threshold, if 0, use RL to predict RUL")

    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.algorithm_name == "rmappo" or all_args.algorithm_name == "rmappg":
        assert (all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy), ("check recurrent policy!")
    elif all_args.algorithm_name == "mappo" or all_args.algorithm_name == "mappg":
        assert (all_args.use_recurrent_policy == False and all_args.use_naive_recurrent_policy == False), ("check recurrent policy!")
    else:
        raise NotImplementedError

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # run dir
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                   0] + "/results") / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # wandb
    if all_args.use_wandb:
        wandb.login()
        run = wandb.init(config=all_args,
                         project="async-RUL",
                         name=str(all_args.algorithm_name) + "_" +
                         str(all_args.experiment_name),
                         job_type="training")
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + \
        str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    num_agents = all_args.num_agents
    all_args.episode_length = all_args.max_steps
    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    # run experiments
    from onpolicy.runner.shared.schedule_runner import ScheduleRunner as Runner

    runner = Runner(config)
    runner.run()
    
    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
