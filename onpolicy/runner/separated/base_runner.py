import os
import time
from itertools import chain

import numpy as np
import torch
import wandb
from tensorboardX import SummaryWriter

from onpolicy.utils.separated_buffer import SeparatedReplayBuffer


def _t2n(x):
    return x.detach().cpu().numpy()


class Runner(object):
    """Separated-actor runner base.

    - One actor per agent (independent parameters)
    - One shared centralized critic (shared parameters + optimizer)

    This is designed to mirror the shared runner structure closely, but uses
    per-agent policies/trainers/buffers.
    """

    def __init__(self, config):
        self.all_args = config["all_args"]
        self.envs = config["envs"]
        self.eval_envs = config["eval_envs"]
        self.device = config["device"]
        self.num_agents = config["num_agents"]

        # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.n_render_rollout_threads = self.all_args.n_render_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.use_wandb = self.all_args.use_wandb
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N

        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        # dir
        self.model_dir = self.all_args.model_dir

        if self.use_render:
            import imageio  # noqa: F401

            self.run_dir = config["run_dir"]
            self.gif_dir = str(self.run_dir / "gifs")
            if not os.path.exists(self.gif_dir):
                os.makedirs(self.gif_dir)
        else:
            if self.use_wandb:
                self.save_dir = str(wandb.run.dir)
                self.run_dir = str(wandb.run.dir)
            else:
                self.run_dir = config["run_dir"]
                self.log_dir = str(self.run_dir / "logs")
                if not os.path.exists(self.log_dir):
                    os.makedirs(self.log_dir)
                self.writter = SummaryWriter(self.log_dir)
                self.save_dir = str(self.run_dir / "models")
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)

        if "mappo" in self.algorithm_name:
            from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO as TrainAlgo
            from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy
        else:
            raise NotImplementedError(
                f"Unsupported algorithm_name={self.algorithm_name} for separated runner in this repo"
            )

        share_observation_space = (
            self.envs.share_observation_space[0]
            if self.use_centralized_V
            else self.envs.observation_space[0]
        )

        # Build per-agent policies.
        self.policy = []
        for agent_id in range(self.num_agents):
            po = Policy(
                self.all_args,
                self.envs.observation_space[0],
                share_observation_space,
                self.envs.action_space[0],
                device=self.device,
            )
            self.policy.append(po)

        # Share critic + critic optimizer across all policies.
        shared_critic = self.policy[0].critic
        shared_critic_optimizer = self.policy[0].critic_optimizer
        for agent_id in range(1, self.num_agents):
            self.policy[agent_id].critic = shared_critic
            self.policy[agent_id].critic_optimizer = shared_critic_optimizer

        if self.model_dir is not None:
            self.restore()

        # algorithm/trainer per agent (actor update per agent; critic shared underneath)
        self.trainer = []
        for agent_id in range(self.num_agents):
            tr = TrainAlgo(self.all_args, self.policy[agent_id], device=self.device)
            self.trainer.append(tr)

        # Share value normalizer across trainers if enabled
        shared_value_norm = getattr(self.trainer[0], "value_normalizer", None)
        shared_policy_value_norm = getattr(self.trainer[0], "policy_value_normalizer", None)
        for agent_id in range(1, self.num_agents):
            if hasattr(self.trainer[agent_id], "value_normalizer"):
                self.trainer[agent_id].value_normalizer = shared_value_norm
            if hasattr(self.trainer[agent_id], "policy_value_normalizer"):
                self.trainer[agent_id].policy_value_normalizer = shared_policy_value_norm

        # buffer per agent
        self.buffer = []
        for agent_id in range(self.num_agents):
            bu = SeparatedReplayBuffer(
                self.all_args,
                self.envs.observation_space[0],
                share_observation_space,
                self.envs.action_space[0],
            )
            self.buffer.append(bu)

    def run(self):
        raise NotImplementedError

    def warmup(self):
        raise NotImplementedError

    def collect(self, step):
        raise NotImplementedError

    def insert(self, data):
        raise NotImplementedError

    @torch.no_grad()
    def compute(self):
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            if isinstance(self.buffer[agent_id].share_obs, dict):
                share_obs_last = self.buffer[agent_id].share_obs["global_obs"][-1]
            else:
                share_obs_last = self.buffer[agent_id].share_obs[-1]

            next_values = self.trainer[agent_id].policy.get_values(
                share_obs_last,
                self.buffer[agent_id].rnn_states_critic[-1],
                self.buffer[agent_id].masks[-1],
            )
            self.buffer[agent_id].compute_returns(_t2n(next_values), self.trainer[agent_id].value_normalizer)

    def train(self):
        train_infos = []
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_training()
            train_info = self.trainer[agent_id].train(self.buffer[agent_id])
            train_infos.append(train_info)
            self.buffer[agent_id].after_update()
        return train_infos

    def save(self):
        # save shared critic once
        critic = self.policy[0].critic
        torch.save(critic.state_dict(), str(self.save_dir) + "/critic.pt")

        # save actors per agent
        for agent_id in range(self.num_agents):
            actor = self.policy[agent_id].actor
            torch.save(actor.state_dict(), str(self.save_dir) + f"/actor_agent{agent_id}.pt")

    def restore(self):
        # restore critic
        critic_state_dict = torch.load(str(self.model_dir) + "/critic.pt", map_location=self.device)
        self.policy[0].critic.load_state_dict(critic_state_dict)

        # restore actors
        for agent_id in range(self.num_agents):
            actor_state_dict = torch.load(
                str(self.model_dir) + f"/actor_agent{agent_id}.pt", map_location=self.device
            )
            self.policy[agent_id].actor.load_state_dict(actor_state_dict)

    def log_train(self, train_infos, total_num_steps):
        for agent_id in range(self.num_agents):
            for k, v in train_infos[agent_id].items():
                agent_k = f"agent{agent_id}/" + k
                if self.use_wandb:
                    wandb.log({agent_k: v}, step=total_num_steps)
                else:
                    self.writter.add_scalars(agent_k, {agent_k: v}, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        for k, v in env_infos.items():
            if len(v) > 0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)
