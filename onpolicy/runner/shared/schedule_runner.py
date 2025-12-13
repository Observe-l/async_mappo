import time
import wandb
import copy
import numpy as np
from onpolicy.utils.util import update_linear_schedule, get_shape_from_act_space
from onpolicy.runner.shared.base_runner import Runner
from onpolicy.envs.rul_schedule.asynccontrol import AsyncControl
import torch
import torch.nn as nn

def _t2n(x):
    return x.detach().cpu().numpy()

class ScheduleRunner(Runner):
    def __init__(self, config):
        super(ScheduleRunner, self).__init__(config)
        self.max_steps = self.all_args.max_steps
        self.async_control = AsyncControl(num_envs=self.n_rollout_threads,num_agents=self.num_agents)
        self.max_rewards = 0
    
    def run(self):
        start = time.time()
        episodes = int(self.num_env_steps) // self.max_steps // self.n_rollout_threads
        for episode in range(episodes):
            values, actions, action_log_probs, rnn_states, rnn_states_critic, action_env = self.warmup()
            self.async_control.reset()
            period_rewards = np.zeros((self.n_rollout_threads, self.num_agents, 1))
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)
            
            while True:
                # Environment step
                dict_obs, rewards, dones, infos = self.envs.step(action_env)
                # Sum up the reward
                period_rewards += rewards
                # Use async_control to indicate whether which agent is active
                self.async_control.step(dict_obs, action_env)
                

                if np.any(self.async_control.active):
                    data = dict_obs, period_rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic
                    self.insert(data, self.async_control.active_agents())
                    # Reset these new actions' rewards to 0
                    for e, a, step in self.async_control.active_agents():
                        period_rewards[e, a, 0] = 0
                    
                    # Compute the new actions, update all the vaiables
                    par_values, par_actions, par_action_log_probs, par_rnn_states, par_rnn_states_critic, action_env = self.async_collect(self.async_control.active_agents())
                    # active_mask = np.array(self.async_control.active == 1).reshape(-1,1)
                    active_mask = (self.async_control.active == 1)
                    values[active_mask] = par_values
                    actions[active_mask] = par_actions
                    action_log_probs[active_mask] = par_action_log_probs
                    rnn_states[active_mask] = par_rnn_states
                    rnn_states_critic[active_mask] = par_rnn_states_critic
                
                '''
                If the envrionment return back done or buffer is full, start a new episode and reset the buffer
                '''
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
            
            self.buffer.update_mask(self.async_control.cnt)
            self.compute()
            train_infos = self.train()

            total_num_steps = (episode + 1) * self.max_steps * self.n_rollout_threads

            
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                      .format(self.all_args.scenario_name,
                              self.algorithm_name,
                              self.experiment_name,
                              episode,
                              episodes,
                              total_num_steps,
                              self.num_env_steps,
                              int(total_num_steps / (end - start))))
                train_infos["average_episode_rewards"] = np.mean(self.buffer.rewards) * self.episode_length
                if self.use_wandb:
                    wandb.log({"average_episode_rewards": train_infos["average_episode_rewards"]})
                print("average episode rewards is {}".format(train_infos["average_episode_rewards"]))

            # train_infos["average_episode_rewards"] = np.mean(self.buffer.rewards) * self.episode_length
            if (episode % self.save_interval == 0 or episode == episodes - 1) and train_infos["average_episode_rewards"] > self.max_rewards:
                self.save()
                self.max_rewards = train_infos["average_episode_rewards"]
                print("Save the model at episode {} with average rewards {}".format(episode, self.max_rewards))
                

    def warmup(self):
        if self.all_args.reset_buffer:
            self.buffer.reset_buffer()
            print("Reset the buffer at the beginning of the episode")
        dict_obs = self.envs.reset() # shape = [num_envs, num_agents, obs_shape]
        # Store the obs for all agents 1 step to update the share_obs
        self.obs_buf = np.zeros((self.n_rollout_threads, self.num_agents, len(dict_obs[0][0])))

        obs, share_obs = self._convert(dict_obs)
        # Init replay buffer
        for key in obs.keys():
            self.buffer.obs[key][0] = obs[key].copy()
        for key in share_obs.keys():
            self.buffer.share_obs[key][0] = share_obs[key].copy()
        
        values, actions, action_log_probs, rnn_states, rnn_states_critic, action_env = self.warm_up_collect(active_agents=self.async_control.active_agents())


        return values, actions, action_log_probs, rnn_states, rnn_states_critic, action_env
    
    def _convert(self, dict_obs):
        '''
        Convert the observation to the right format for async buffer
        Input: dict_obs = {agent_id: obs}
        Ountput: obs = {"global_obs": obs}
        share_obs = {"global_obs": share_obs}

        This function will combine new obs from activate agents and old obs from inactive agents
        '''
        obs = {}
        obs["global_obs"] = np.zeros((len(dict_obs), self.num_agents, *self.envs.observation_space[0]["global_obs"].shape))
        for e, a, step in self.async_control.active_agents():
            obs["global_obs"][e, a] = dict_obs[e][a].copy()
            # obs_list = list(dict_obs[e].values())
            self.obs_buf[e, a] = dict_obs[e][a].copy()
        share_obs = self.obs_buf.reshape(self.n_rollout_threads, -1)
        share_obs = {"global_obs": np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)}

        return obs, share_obs

    @torch.no_grad()
    def warm_up_collect(self, active_agents):
        '''
        Collect the data for warmup
        '''
        self.trainer.prep_rollout()
        for key in self.buffer.share_obs.keys():
            concat_share_obs = np.concatenate(self.buffer.share_obs[key][0])
        for key in self.buffer.obs.keys():
            concat_obs = np.concatenate(self.buffer.obs[key][0])

        value, action, action_log_prob, rnn_states, rnn_states_critic \
            = self.trainer.policy.get_actions(concat_share_obs,
                                              concat_obs,
                                              np.concatenate(self.buffer.rnn_states[0]),
                                              np.concatenate(self.buffer.rnn_states_critic[0]),
                                              np.concatenate(self.buffer.masks[0]))
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))

        # Generate the action for activate agents from actions
        action_env = [{} for _ in range(self.n_rollout_threads)]
        for e, a, step in active_agents:
            action_env[e][a] = actions[e][a][0].copy()
        
        return values, actions, action_log_probs, rnn_states, rnn_states_critic, action_env

    @torch.no_grad()
    def async_collect(self, active_agents):
        '''
        
        '''
        self.trainer.prep_rollout()

        # concat_share_obs = {}
        # concat_obs = {}

        for key in self.buffer.share_obs.keys():
            concat_share_obs = np.stack([self.buffer.share_obs[key][step, e, a] for e, a, step in active_agents], axis=0)
        for key in self.buffer.obs.keys():
            concat_obs = np.stack([self.buffer.obs[key][step, e, a] for e, a, step in active_agents], axis=0)

        value, action, action_log_prob, rnn_states, rnn_states_critic \
            = self.trainer.policy.get_actions(concat_share_obs,
                                              concat_obs,
                                              np.stack([self.buffer.rnn_states[step, e, a] for e, a, step in active_agents], axis=0),
                                              np.stack([self.buffer.rnn_states_critic[step, e, a] for e, a, step in active_agents], axis=0),
                                              np.stack([self.buffer.masks[step, e, a] for e, a, step in active_agents], axis=0))
        # [self.envs, agents, dim]
        values = _t2n(value)
        actions = _t2n(action)
        action_log_probs = _t2n(action_log_prob)
        rnn_states = _t2n(rnn_states)
        rnn_states_critic = _t2n(rnn_states_critic)

        # Generate the action for activate agents from actions
        action_env = [{} for _ in range(self.n_rollout_threads)]
        for i, (e, a, step) in enumerate(active_agents):
            action_env[e][a] = actions[i,0].copy()
        
        return values, actions, action_log_probs, rnn_states, rnn_states_critic, action_env

    @torch.no_grad()
    def compute(self):
        self.trainer.prep_rollout()
        # concat_share_obs = {}
        for key in self.buffer.share_obs.keys():
            concat_share_obs = np.concatenate(self.buffer.share_obs[key][-1])

        next_values = self.trainer.policy.get_values(concat_share_obs,
                                                     np.concatenate(
                                                         self.buffer.rnn_states_critic[-1]),
                                                     np.concatenate(self.buffer.masks[-1]))
        next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)

    def insert(self, data, active_agents):
        dict_obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data
        dones_env = np.all(dones, axis=-1)

        rnn_states[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(
        ), self.num_agents, *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)
        
        obs, share_obs = self._convert(dict_obs)

        self.buffer.async_insert(share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs, values, rewards, masks, active_agents=active_agents)