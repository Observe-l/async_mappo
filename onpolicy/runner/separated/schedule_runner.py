import time

import numpy as np
import torch
import wandb

from onpolicy.envs.rul_schedule.asynccontrol import AsyncControl
from onpolicy.runner.separated.base_runner import Runner
from onpolicy.utils.util import update_linear_schedule


def _t2n(x):
    return x.detach().cpu().numpy()


class ScheduleRunner(Runner):
    def __init__(self, config):
        super(ScheduleRunner, self).__init__(config)
        self.max_steps = self.all_args.max_steps
        self.async_control = AsyncControl(num_envs=self.n_rollout_threads, num_agents=self.num_agents)
        self.max_rewards = -1e9

        # rolling buffer of last obs per (env, agent)
        # initialized on first reset in warmup
        self.obs_buf = None

    def run(self):
        start = time.time()
        episodes = int(self.num_env_steps) // self.max_steps // self.n_rollout_threads

        for episode in range(episodes):
            values, actions, action_log_probs, rnn_states, rnn_states_critic, action_env = self.warmup()
            self.async_control.reset()
            period_rewards = np.zeros((self.n_rollout_threads, self.num_agents, 1))

            if self.use_linear_lr_decay:
                # Decay each actor LR, but decay the shared critic LR only once.
                for agent_id in range(self.num_agents):
                    update_linear_schedule(
                        self.policy[agent_id].actor_optimizer,
                        episode,
                        episodes,
                        self.policy[agent_id].lr,
                    )
                update_linear_schedule(
                    self.policy[0].critic_optimizer,
                    episode,
                    episodes,
                    self.policy[0].critic_lr,
                )

            while True:
                dict_obs, rewards, dones, infos = self.envs.step(action_env)
                period_rewards += rewards

                self.async_control.step(dict_obs, action_env)

                if np.any(self.async_control.active):
                    data = (
                        dict_obs,
                        period_rewards,
                        dones,
                        infos,
                        values,
                        actions,
                        action_log_probs,
                        rnn_states,
                        rnn_states_critic,
                    )
                    active_agents = self.async_control.active_agents()
                    self.insert(data, active_agents)

                    # reset rewards for those agents that just acted
                    for e, a, step in active_agents:
                        period_rewards[e, a, 0] = 0

                    par_values, par_actions, par_action_log_probs, par_rnn_states, par_rnn_states_critic, action_env = (
                        self.async_collect(active_agents)
                    )

                    active_mask = self.async_control.active == 1
                    values[active_mask] = par_values
                    actions[active_mask] = par_actions
                    action_log_probs[active_mask] = par_action_log_probs
                    rnn_states[active_mask] = par_rnn_states
                    rnn_states_critic[active_mask] = par_rnn_states_critic

                done_flag = False
                for env_done in dones:
                    if 'bool' in env_done.__class__.__name__:
                        if env_done:
                            done_flag = True
                    else:
                        if np.all(env_done):
                            done_flag = True

                max_step = np.max(self.async_control.cnt)
                if (max_step >= self.max_steps) or done_flag:
                    break

            # per-agent mask update
            for agent_id in range(self.num_agents):
                self.buffer[agent_id].update_mask(self.async_control.cnt[:, agent_id])

            self.compute()
            train_infos = self.train()

            total_num_steps = (episode + 1) * self.max_steps * self.n_rollout_threads

            if episode % self.log_interval == 0:
                end = time.time()
                print(
                    "\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n".format(
                        self.all_args.scenario_name,
                        self.algorithm_name,
                        self.experiment_name,
                        episode,
                        episodes,
                        total_num_steps,
                        self.num_env_steps,
                        int(total_num_steps / (end - start)),
                    )
                )

                # use agent-averaged rewards for reporting
                avg_ep_rew = np.mean([np.mean(self.buffer[a].rewards) for a in range(self.num_agents)]) * self.episode_length
                if self.use_wandb:
                    wandb.log({"average_episode_rewards": avg_ep_rew}, step=total_num_steps)
                print(f"average episode rewards is {avg_ep_rew}")

            # save on improvement
            avg_ep_rew = np.mean([np.mean(self.buffer[a].rewards) for a in range(self.num_agents)]) * self.episode_length
            if (episode % self.save_interval == 0 or episode == episodes - 1) and avg_ep_rew > self.max_rewards:
                self.save()
                self.max_rewards = avg_ep_rew
                print(f"Save the model at episode {episode} with average rewards {self.max_rewards}")

    def warmup(self):
        if getattr(self.all_args, "reset_buffer", False):
            for agent_id in range(self.num_agents):
                self.buffer[agent_id].reset_buffer()
            print("Reset the buffer at the beginning of the episode")

        dict_obs = self.envs.reset()

        obs_dim = len(dict_obs[0][0])
        if self.obs_buf is None:
            self.obs_buf = np.zeros((self.n_rollout_threads, self.num_agents, obs_dim), dtype=np.float32)

        obs_list, share_obs = self._convert(dict_obs)

        # init replay buffer first obs
        for agent_id in range(self.num_agents):
            for key in obs_list[agent_id].keys():
                self.buffer[agent_id].obs[key][0] = obs_list[agent_id][key].copy()
            for key in share_obs.keys():
                self.buffer[agent_id].share_obs[key][0] = share_obs[key].copy()

        values, actions, action_log_probs, rnn_states, rnn_states_critic, action_env = self.warm_up_collect(
            active_agents=self.async_control.active_agents()
        )

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, action_env

    def _convert(self, dict_obs):
        """Convert env dict obs into per-agent obs + shared global obs.

        obs_list[agent_id] => {"global_obs": (n_envs, obs_dim)}
        share_obs => {"global_obs": (n_envs, num_agents*obs_dim)}

        Keeps stale obs for inactive agents using self.obs_buf.
        """
        for e, a, step in self.async_control.active_agents():
            self.obs_buf[e, a] = dict_obs[e][a].copy()

        share_obs = self.obs_buf.reshape(self.n_rollout_threads, -1)
        share_obs = {"global_obs": share_obs}

        obs_list = []
        for agent_id in range(self.num_agents):
            obs_list.append({"global_obs": self.obs_buf[:, agent_id].copy()})

        return obs_list, share_obs

    @torch.no_grad()
    def warm_up_collect(self, active_agents):
        # compute actions for all agents at reset
        values = None
        actions = None
        action_log_probs = None
        rnn_states = None
        rnn_states_critic = None

        action_env = [{} for _ in range(self.n_rollout_threads)]

        # prepare shared obs (same for all agents)
        share_obs = self.buffer[0].share_obs["global_obs"][0]

        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()

            obs = self.buffer[agent_id].obs["global_obs"][0]
            value, action, action_log_prob, rs, rsc = self.trainer[agent_id].policy.get_actions(
                share_obs,
                obs,
                self.buffer[agent_id].rnn_states[0],
                self.buffer[agent_id].rnn_states_critic[0],
                self.buffer[agent_id].masks[0],
            )

            v = _t2n(value)
            a = _t2n(action)
            alp = _t2n(action_log_prob)
            rs = _t2n(rs)
            rsc = _t2n(rsc)

            if values is None:
                values = np.zeros((self.n_rollout_threads, self.num_agents, *v.shape[1:]), dtype=np.float32)
                actions = np.zeros((self.n_rollout_threads, self.num_agents, *a.shape[1:]), dtype=np.float32)
                action_log_probs = np.zeros(
                    (self.n_rollout_threads, self.num_agents, *alp.shape[1:]), dtype=np.float32
                )
                rnn_states = np.zeros(
                    (self.n_rollout_threads, self.num_agents, *rs.shape[1:]), dtype=np.float32
                )
                rnn_states_critic = np.zeros(
                    (self.n_rollout_threads, self.num_agents, *rsc.shape[1:]), dtype=np.float32
                )

            values[:, agent_id] = v
            actions[:, agent_id] = a
            action_log_probs[:, agent_id] = alp
            rnn_states[:, agent_id] = rs
            rnn_states_critic[:, agent_id] = rsc

            # fill env action dicts
            for e in range(self.n_rollout_threads):
                action_env[e][agent_id] = actions[e, agent_id][0].copy()

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, action_env

    @torch.no_grad()
    def async_collect(self, active_agents):
        # compute actions only for active agents
        n_active = len(active_agents)

        # output arrays aligned with active_agents order
        par_values = np.zeros((n_active, 1), dtype=np.float32)
        par_actions = None
        par_action_log_probs = None
        par_rnn_states = None
        par_rnn_states_critic = None

        # group indices by agent
        by_agent = {}
        for i, (e, a, step) in enumerate(active_agents):
            by_agent.setdefault(a, []).append(i)

        # shared obs is identical across agents; we sample from each agent buffer for simplicity
        action_env = [{} for _ in range(self.n_rollout_threads)]

        for agent_id, idxs in by_agent.items():
            self.trainer[agent_id].prep_rollout()

            env_ids = [active_agents[i][0] for i in idxs]
            steps = [active_agents[i][2] for i in idxs]

            share_obs = np.stack(
                [self.buffer[agent_id].share_obs["global_obs"][step, e] for e, step in zip(env_ids, steps)],
                axis=0,
            )
            obs = np.stack(
                [self.buffer[agent_id].obs["global_obs"][step, e] for e, step in zip(env_ids, steps)],
                axis=0,
            )

            rs = np.stack(
                [self.buffer[agent_id].rnn_states[step, e] for e, step in zip(env_ids, steps)], axis=0
            )
            rsc = np.stack(
                [self.buffer[agent_id].rnn_states_critic[step, e] for e, step in zip(env_ids, steps)],
                axis=0,
            )
            masks = np.stack(
                [self.buffer[agent_id].masks[step, e] for e, step in zip(env_ids, steps)], axis=0
            )

            value, action, action_log_prob, rs2, rsc2 = self.trainer[agent_id].policy.get_actions(
                share_obs,
                obs,
                rs,
                rsc,
                masks,
            )

            v = _t2n(value)
            a = _t2n(action)
            alp = _t2n(action_log_prob)
            rs2 = _t2n(rs2)
            rsc2 = _t2n(rsc2)

            if par_actions is None:
                par_actions = np.zeros((n_active, *a.shape[1:]), dtype=np.float32)
                par_action_log_probs = np.zeros((n_active, *alp.shape[1:]), dtype=np.float32)
                par_rnn_states = np.zeros((n_active, *rs2.shape[1:]), dtype=np.float32)
                par_rnn_states_critic = np.zeros((n_active, *rsc2.shape[1:]), dtype=np.float32)

            for local_j, global_i in enumerate(idxs):
                par_values[global_i] = v[local_j]
                par_actions[global_i] = a[local_j]
                par_action_log_probs[global_i] = alp[local_j]
                par_rnn_states[global_i] = rs2[local_j]
                par_rnn_states_critic[global_i] = rsc2[local_j]

                e, a_id, _ = active_agents[global_i]
                action_env[e][a_id] = par_actions[global_i][0].copy()

        return par_values, par_actions, par_action_log_probs, par_rnn_states, par_rnn_states_critic, action_env

    def insert(self, data, active_agents):
        dict_obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data
        dones_env = np.all(dones, axis=-1)

        rnn_states[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32
        )
        rnn_states_critic[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32
        )

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        obs_list, share_obs = self._convert(dict_obs)

        # per-agent async insert
        for agent_id in range(self.num_agents):
            active_for_agent = [(e, a, step) for (e, a, step) in active_agents if a == agent_id]
            if len(active_for_agent) == 0:
                continue

            self.buffer[agent_id].async_insert(
                share_obs,
                obs_list[agent_id],
                rnn_states[:, agent_id],
                rnn_states_critic[:, agent_id],
                actions[:, agent_id],
                action_log_probs[:, agent_id],
                values[:, agent_id],
                rewards[:, agent_id],
                masks[:, agent_id],
                active_envs=active_for_agent,
            )
