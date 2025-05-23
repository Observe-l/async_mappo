import torch
import numpy as np
from collections import defaultdict

from onpolicy.utils.util import check,get_shape_from_obs_space, get_shape_from_act_space

def _flatten(T, N, x):
    return x.reshape(T * N, *x.shape[2:])

def _cast(x):
    return x.transpose(1, 2, 0, 3).reshape(-1, *x.shape[3:])

class SharedReplayBuffer(object):
    def __init__(self, args, num_agents, obs_space, share_obs_space, act_space):
        self.episode_length = args.episode_length
        self.num_agents = num_agents
        self.n_rollout_threads = args.n_rollout_threads
        self.hidden_size = args.hidden_size
        self.recurrent_N = args.recurrent_N
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self._use_gae = args.use_gae
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_proper_time_limits = args.use_proper_time_limits 
        self.asynch = args.asynch

        self._mixed_obs = False  # for mixed observation   

        obs_shape = get_shape_from_obs_space(obs_space)
        share_obs_shape = get_shape_from_obs_space(share_obs_space)

        # for mixed observation
        if 'dict' in obs_shape.__class__.__name__:
            self._mixed_obs = True
            
            self.obs = {}
            self.share_obs = {}

            for key in obs_shape:
                self.obs[key] = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, *obs_shape[key].shape), dtype=np.float32)
            for key in share_obs_shape:
                self.share_obs[key] = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, *share_obs_shape[key].shape), dtype=np.float32)
        
        else: 
            # deal with special attn format   
            if type(obs_shape[-1]) == list:
                obs_shape = obs_shape[:1]

            if type(share_obs_shape[-1]) == list:
                share_obs_shape = share_obs_shape[:1]

            self.share_obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, *share_obs_shape), dtype=np.float32)
            self.obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, *obs_shape), dtype=np.float32)

        self.rnn_states = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        self.rnn_states_critic = np.zeros_like(self.rnn_states)
       
        self.value_preds = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, num_agents, 1), dtype=np.float32)
        self.returns = np.zeros_like(self.value_preds)
               
        if act_space.__class__.__name__ == 'Discrete':
            self.available_actions = np.ones((self.episode_length + 1, self.n_rollout_threads, num_agents, act_space.n), dtype=np.float32)
        else:
            self.available_actions = None

        act_shape = get_shape_from_act_space(act_space)
        if args.grid_goal:
            act_shape=3

        self.actions = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, act_shape), dtype=np.float32)
        self.action_log_probs = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, act_shape), dtype=np.float32)
        self.rewards = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, 1), dtype=np.float32)

        self.masks = np.ones((self.episode_length + 1, self.n_rollout_threads, num_agents, 1), dtype=np.float32)
        self.bad_masks = np.ones_like(self.masks)
        self.active_masks = np.ones_like(self.masks)

        self.update_step = np.zeros((self.n_rollout_threads, num_agents, 1), dtype=np.int32)

        self.step = 0

    def insert(self, share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs,
               value_preds, rewards, masks, bad_masks=None, active_masks=None, available_actions=None):

        if self._mixed_obs:
            for key in self.share_obs.keys():
                self.share_obs[key][self.step + 1] = share_obs[key].copy()
            for key in self.obs.keys():
                self.obs[key][self.step + 1] = obs[key].copy()
        else:
            self.share_obs[self.step + 1] = share_obs.copy()
            self.obs[self.step + 1] = obs.copy()

        self.rnn_states[self.step + 1] = rnn_states.copy()
        self.rnn_states_critic[self.step + 1] = rnn_states_critic.copy()
        self.actions[self.step] = actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.rewards[self.step] = rewards.copy()
        self.masks[self.step + 1] = masks.copy()
        if bad_masks is not None:
            self.bad_masks[self.step + 1] = bad_masks.copy()
        if active_masks is not None:
            self.active_masks[self.step + 1] = active_masks.copy()
        if available_actions is not None:
            self.available_actions[self.step + 1] = available_actions.copy()

        self.step = (self.step + 1) % self.episode_length
    
    def async_insert(self, share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs,
               value_preds, rewards, masks, bad_masks=None, active_masks=None, available_actions=None, active_agents=None):
        assert active_agents is not None
        for e, a, step in active_agents:
            step = int(step - 1)
            if step == -1:
                step = self.update_step[e, a]
                if self._mixed_obs:
                    for key in self.share_obs.keys():
                        self.share_obs[key][step + 1, e, a] = share_obs[key][e, a].copy()
                    for key in self.obs.keys():
                        self.obs[key][step + 1, e, a] = obs[key][e, a].copy()
                else:
                    self.share_obs[step + 1, e, a] = share_obs[e, a].copy()
                    self.obs[step + 1, e, a] = obs[e, a].copy()

                self.rnn_states[step + 1, e, a] = rnn_states[e, a].copy()
                self.rnn_states_critic[step + 1, e, a] = rnn_states_critic[e, a].copy()
                self.masks[step + 1] = masks[e, a].copy()
                if bad_masks is not None:
                    self.bad_masks[step + 1] = bad_masks[e, a].copy()
                if active_masks is not None:
                    self.active_masks[step + 1] = active_masks[e, a].copy()
                if available_actions is not None:
                    self.available_actions[step + 1] = available_actions[e, a].copy()
                continue
            if self._mixed_obs:
                for key in self.share_obs.keys():
                    self.share_obs[key][step + 1, e, a] = share_obs[key][e, a].copy()
                for key in self.obs.keys():
                    self.obs[key][step + 1, e, a] = obs[key][e, a].copy()
            else:
                self.share_obs[step + 1, e, a] = share_obs[e, a].copy()
                self.obs[step + 1, e, a] = obs[e, a].copy()

            self.rnn_states[step + 1, e, a] = rnn_states[e, a].copy()
            self.rnn_states_critic[step + 1, e, a] = rnn_states_critic[e, a].copy()
            self.actions[step, e, a] = actions[e, a].copy()
            self.action_log_probs[step, e, a] = action_log_probs[e, a].copy()
            self.value_preds[step, e, a] = value_preds[e, a].copy()
            self.rewards[step, e, a] = rewards[e, a].copy()
            self.masks[step + 1] = masks[e, a].copy()
            if bad_masks is not None:
                self.bad_masks[step + 1] = bad_masks[e, a].copy()
            if active_masks is not None:
                self.active_masks[step + 1] = active_masks[e, a].copy()
            if available_actions is not None:
                self.available_actions[step + 1] = available_actions[e, a].copy()
            
            self.update_step[e, a] = step

    def update_mask(self, steps):
        self.active_masks = np.ones_like(self.active_masks)
        for e in range(self.n_rollout_threads):
            for a in range(self.num_agents):
                step = self.update_step[e, a, 0]
                self.masks[step+1:, e, a] = np.zeros_like(self.masks[step+1:, e, a])
                self.active_masks[step+1:, e, a] = np.zeros_like(self.active_masks[step+1:, e, a])

    def chooseinsert(self, share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs,
                     value_preds, rewards, masks, bad_masks=None, active_masks=None, available_actions=None):
        self.share_obs[self.step] = share_obs.copy()
        self.obs[self.step] = obs.copy()
        self.rnn_states[self.step + 1] = rnn_states.copy()
        self.rnn_states_critic[self.step + 1] = rnn_states_critic.copy()
        self.actions[self.step] = actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.rewards[self.step] = rewards.copy()
        self.masks[self.step + 1] = masks.copy()
        if bad_masks is not None:
            self.bad_masks[self.step + 1] = bad_masks.copy()
        if active_masks is not None:
            self.active_masks[self.step] = active_masks.copy()
        if available_actions is not None:
            self.available_actions[self.step] = available_actions.copy()

        self.step = (self.step + 1) % self.episode_length

    def after_update(self):
        if self.asynch:
            for e in range(self.n_rollout_threads):
                for a in range(self.num_agents):
                    step = self.update_step[e, a] + 1
                    if self._mixed_obs:
                        for key in self.share_obs.keys():
                            self.share_obs[key][0, e, a] = self.share_obs[key][step, e, a].copy()
                        for key in self.obs.keys():
                            self.obs[key][0, e, a] = self.obs[key][step, e, a].copy()
                    else:
                        self.share_obs[0, e, a] = self.share_obs[step, e, a].copy()
                        self.obs[0, e, a] = self.obs[step, e, a].copy()
                    self.rnn_states[0, e, a] = self.rnn_states[step, e, a].copy()
                    self.rnn_states_critic[0, e, a] = self.rnn_states_critic[step, e, a].copy()
                    self.masks[0, e, a] = self.masks[step, e, a].copy()
                    self.bad_masks[0, e, a] = self.bad_masks[step, e, a].copy()
                    self.active_masks[0, e, a] = self.active_masks[step, e, a].copy()
                    if self.available_actions is not None:
                        self.available_actions[0, e, a] = self.available_actions[step, e, a].copy()
        else:
            if self._mixed_obs:
                for key in self.share_obs.keys():
                    self.share_obs[key][0] = self.share_obs[key][-1].copy()
                for key in self.obs.keys():
                    self.obs[key][0] = self.obs[key][-1].copy()
            else:
                self.share_obs[0] = self.share_obs[-1].copy()
                self.obs[0] = self.obs[-1].copy()
            self.rnn_states[0] = self.rnn_states[-1].copy()
            self.rnn_states_critic[0] = self.rnn_states_critic[-1].copy()
            self.masks[0] = self.masks[-1].copy()
            self.bad_masks[0] = self.bad_masks[-1].copy()
            self.active_masks[0] = self.active_masks[-1].copy()
            if self.available_actions is not None:
                self.available_actions[0] = self.available_actions[-1].copy()
    
    def chooseafter_update(self):
        self.rnn_states[0] = self.rnn_states[-1].copy()
        self.rnn_states_critic[0] = self.rnn_states_critic[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()

    def compute_returns(self, next_value, value_normalizer=None):
        if self._use_proper_time_limits:
            if self._use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        # step + 1
                        delta = self.rewards[step] + self.gamma * value_normalizer.denormalize(self.value_preds[step + 1]) * self.masks[step + 1]  \
                            - value_normalizer.denormalize(self.value_preds[step])
                        gae = delta + self.gamma * self.gae_lambda * gae * self.masks[step + 1] 
                        gae = gae * self.bad_masks[step + 1]
                        self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])
                    else:
                        delta = self.rewards[step] + self.gamma * self.value_preds[step + 1] * self.masks[step + 1] - self.value_preds[step] 
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae 
                        gae = gae * self.bad_masks[step + 1]
                        self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        self.returns[step] = (self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[step]) * self.bad_masks[step + 1] \
                                            + (1 - self.bad_masks[step + 1]) * value_normalizer.denormalize(self.value_preds[step]) 
                    else:
                        self.returns[step] = (self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[step]) * self.bad_masks[step + 1] \
                                            + (1 - self.bad_masks[step + 1]) * self.value_preds[step]
        else:
            if self._use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        delta = self.rewards[step] + self.gamma * value_normalizer.denormalize(self.value_preds[step + 1]) * self.masks[step + 1] \
                            - value_normalizer.denormalize(self.value_preds[step])
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae 
                        self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step]) 
                    else:
                        delta = self.rewards[step] + self.gamma * self.value_preds[step + 1] * self.masks[step + 1] - self.value_preds[step] 
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae 
                        self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.shape[0])):
                    self.returns[step] = self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[step]

    def feed_forward_generator(self, advantages, num_mini_batch=None, mini_batch_size=None):
        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        batch_size = n_rollout_threads * episode_length * num_agents

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) * number of agents ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(n_rollout_threads, episode_length, num_agents, n_rollout_threads * episode_length * num_agents,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch

        rand = torch.randperm(batch_size).numpy()
        sampler = [rand[i*mini_batch_size:(i+1)*mini_batch_size] for i in range(num_mini_batch)]

        if self._mixed_obs:
            share_obs = {}
            obs = {}
            for key in self.share_obs.keys():
                share_obs[key] = self.share_obs[key][:-1].reshape(-1, *self.share_obs[key].shape[3:])
            for key in self.obs.keys():
                obs[key] = self.obs[key][:-1].reshape(-1, *self.obs[key].shape[3:])
        else:
            share_obs = self.share_obs[:-1].reshape(-1, *self.share_obs.shape[3:])
            obs = self.obs[:-1].reshape(-1, *self.obs.shape[3:])
        rnn_states = self.rnn_states[:-1].reshape(-1, *self.rnn_states.shape[3:])
        rnn_states_critic = self.rnn_states_critic[:-1].reshape(-1, *self.rnn_states_critic.shape[3:])
        actions = self.actions.reshape(-1, self.actions.shape[-1])
        if self.available_actions is not None:
            available_actions = self.available_actions[:-1].reshape(-1, self.available_actions.shape[-1])
        value_preds = self.value_preds[:-1].reshape(-1, 1)
        returns = self.returns[:-1].reshape(-1, 1)
        masks = self.masks[:-1].reshape(-1, 1)
        active_masks = self.active_masks[:-1].reshape(-1, 1)
        action_log_probs = self.action_log_probs.reshape(-1, self.action_log_probs.shape[-1])
        advantages = advantages.reshape(-1, 1)

        for indices in sampler:
            # obs size [T+1 N M Dim]-->[T N M Dim]-->[T*N*M,Dim]-->[index,Dim]
            if self._mixed_obs:
                share_obs_batch = {}
                obs_batch = {}
                for key in share_obs.keys():
                    share_obs_batch[key] = share_obs[key][indices]
                for key in obs.keys():
                    obs_batch[key] = obs[key][indices]
            else:
                share_obs_batch = share_obs[indices]
                obs_batch = obs[indices]
            rnn_states_batch = rnn_states[indices]
            rnn_states_critic_batch = rnn_states_critic[indices]
            actions_batch = actions[indices]
            if self.available_actions is not None:
                available_actions_batch = available_actions[indices]
            else:
                available_actions_batch = None
            value_preds_batch = value_preds[indices]
            return_batch = returns[indices]
            masks_batch = masks[indices]
            active_masks_batch = active_masks[indices]
            old_action_log_probs_batch = action_log_probs[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages[indices]

            yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, adv_targ, available_actions_batch

    def naive_recurrent_generator(self, advantages, num_mini_batch):
        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        batch_size = n_rollout_threads*num_agents
        assert n_rollout_threads*num_agents >= num_mini_batch, (
            "PPO requires the number of processes ({})* number of agents ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(n_rollout_threads, num_agents, num_mini_batch))
        num_envs_per_batch = batch_size // num_mini_batch
        perm = torch.randperm(batch_size).numpy()
        
        if self._mixed_obs:
            share_obs = {}
            obs = {}
            for key in self.share_obs.keys():
                share_obs[key] = self.share_obs[key].reshape(-1, batch_size, *self.share_obs[key].shape[3:])
            for key in self.obs.keys():
                obs[key] = self.obs[key].reshape(-1, batch_size, *self.obs[key].shape[3:])
        else:
            share_obs = self.share_obs.reshape(-1, batch_size, *self.share_obs.shape[3:])
            obs = self.obs.reshape(-1, batch_size, *self.obs.shape[3:])
        rnn_states = self.rnn_states.reshape(-1, batch_size, *self.rnn_states.shape[3:])
        rnn_states_critic = self.rnn_states_critic.reshape(-1, batch_size, *self.rnn_states_critic.shape[3:])
        actions = self.actions.reshape(-1, batch_size, self.actions.shape[-1])
        if self.available_actions is not None:
            available_actions = self.available_actions.reshape(-1, batch_size, self.available_actions.shape[-1])
        value_preds = self.value_preds.reshape(-1, batch_size, 1)
        returns = self.returns.reshape(-1, batch_size, 1)
        masks = self.masks.reshape(-1, batch_size, 1)
        active_masks = self.active_masks.reshape(-1, batch_size, 1)
        action_log_probs = self.action_log_probs.reshape(-1, batch_size, self.action_log_probs.shape[-1])
        advantages = advantages.reshape(-1, batch_size, 1)

        for start_ind in range(0, batch_size, num_envs_per_batch):

            if self._mixed_obs:
                share_obs_batch = defaultdict(list)
                obs_batch = defaultdict(list)
            else:
                share_obs_batch = []
                obs_batch = []

            rnn_states_batch = []
            rnn_states_critic_batch = []
            actions_batch = []
            available_actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            active_masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                if self._mixed_obs:
                    for key in share_obs.keys():
                        share_obs_batch[key].append(share_obs[key][:-1, ind])
                    for key in obs.keys():
                        obs_batch[key].append(obs[key][:-1, ind])
                else:
                    share_obs_batch.append(share_obs[:-1, ind])
                    obs_batch.append(obs[:-1, ind])
                rnn_states_batch.append(rnn_states[0:1, ind])
                rnn_states_critic_batch.append(rnn_states_critic[0:1, ind])
                actions_batch.append(actions[:, ind])
                if self.available_actions is not None:
                    available_actions_batch.append(available_actions[:-1, ind])
                value_preds_batch.append(value_preds[:-1, ind])
                return_batch.append(returns[:-1, ind])
                masks_batch.append(masks[:-1, ind])
                active_masks_batch.append(active_masks[:-1, ind])
                old_action_log_probs_batch.append(action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])
            
            # [N[T, dim]]
            T, N = self.episode_length, num_envs_per_batch
            # These are all from_numpys of size (T, N, -1)
            if self._mixed_obs:
                for key in share_obs_batch.keys():
                    share_obs_batch[key] = np.stack(share_obs_batch[key], 1)
                for key in obs_batch.keys():
                    obs_batch[key] = np.stack(obs_batch[key], 1)
            else:
                share_obs_batch = np.stack(share_obs_batch, 1)
                obs_batch = np.stack(obs_batch, 1)
            actions_batch = np.stack(actions_batch, 1)
            if self.available_actions is not None:
                available_actions_batch = np.stack(available_actions_batch, 1)
            value_preds_batch = np.stack(value_preds_batch, 1)
            return_batch = np.stack(return_batch, 1)
            masks_batch = np.stack(masks_batch, 1)
            active_masks_batch = np.stack(active_masks_batch, 1)
            old_action_log_probs_batch = np.stack(old_action_log_probs_batch, 1)
            adv_targ = np.stack(adv_targ, 1)

            # States is just a (N, dim) from_numpy [N[1,dim]]
            rnn_states_batch = np.stack(rnn_states_batch).reshape(N, *self.rnn_states.shape[3:])
            rnn_states_critic_batch = np.stack(rnn_states_critic_batch).reshape(N, *self.rnn_states_critic.shape[3:])

            # Flatten the (T, N, ...) from_numpys to (T * N, ...)
            if self._mixed_obs:
                for key in share_obs_batch.keys():
                    share_obs_batch[key] = _flatten(T, N, share_obs_batch[key])
                for key in obs_batch.keys():
                    obs_batch[key] = _flatten(T, N, obs_batch[key])
            else:
                share_obs_batch = _flatten(T, N, share_obs_batch)
                obs_batch = _flatten(T, N, obs_batch)
            actions_batch = _flatten(T, N, actions_batch)
            if self.available_actions is not None:
                available_actions_batch = _flatten(T, N, available_actions_batch)
            else:
                available_actions_batch = None
            value_preds_batch = _flatten(T, N, value_preds_batch)
            return_batch = _flatten(T, N, return_batch)
            masks_batch = _flatten(T, N, masks_batch)
            active_masks_batch = _flatten(T, N, active_masks_batch)
            old_action_log_probs_batch = _flatten(T, N, old_action_log_probs_batch)
            adv_targ = _flatten(T, N, adv_targ)

            yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, adv_targ, available_actions_batch

    def recurrent_generator(self, advantages, num_mini_batch, data_chunk_length):
        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        batch_size = n_rollout_threads * episode_length * num_agents
        data_chunks = batch_size // data_chunk_length  # [C=r*T*M/L]
        mini_batch_size = data_chunks // num_mini_batch

        assert n_rollout_threads * episode_length * num_agents >= data_chunk_length, (
            "PPO requires the number of processes ({})* number of agents ({}) * episode length ({}) "
            "to be greater than or equal to the number of "
            "data chunk length ({}).".format(n_rollout_threads, num_agents, episode_length ,data_chunk_length))

        rand = torch.randperm(data_chunks).numpy()
        sampler = [rand[i*mini_batch_size:(i+1)*mini_batch_size] for i in range(num_mini_batch)]

        if self._mixed_obs:
            share_obs = {}
            obs = {}
            for key in self.share_obs.keys():
                if len(self.share_obs[key].shape) == 6:
                    share_obs[key] = self.share_obs[key][:-1].transpose(1, 2, 0, 3, 4, 5).reshape(-1, *self.share_obs[key].shape[3:])
                elif len(self.share_obs[key].shape) == 5:
                    share_obs[key] = self.share_obs[key][:-1].transpose(1, 2, 0, 3, 4).reshape(-1, *self.share_obs[key].shape[3:])
                else:
                    share_obs[key] = _cast(self.share_obs[key][:-1])
                   
            for key in self.obs.keys():
                if len(self.obs[key].shape) == 6:
                    obs[key] = self.obs[key][:-1].transpose(1, 2, 0, 3, 4, 5).reshape(-1, *self.obs[key].shape[3:])
                elif len(self.obs[key].shape) == 5:
                    obs[key] = self.obs[key][:-1].transpose(1, 2, 0, 3, 4).reshape(-1, *self.obs[key].shape[3:])
                else:
                    obs[key] = _cast(self.obs[key][:-1])
        else:
            if len(self.share_obs.shape) > 4:
                share_obs = self.share_obs[:-1].transpose(1, 2, 0, 3, 4, 5).reshape(-1, *self.share_obs.shape[3:])
                obs = self.obs[:-1].transpose(1, 2, 0, 3, 4, 5).reshape(-1, *self.obs.shape[3:])
            else:
                share_obs = _cast(self.share_obs[:-1])
                obs = _cast(self.obs[:-1])

        actions = _cast(self.actions)
        action_log_probs = _cast(self.action_log_probs)
        advantages = _cast(advantages)
        value_preds = _cast(self.value_preds[:-1])
        returns = _cast(self.returns[:-1])
        masks = _cast(self.masks[:-1])
        active_masks = _cast(self.active_masks[:-1])       
        # rnn_states = _cast(self.rnn_states[:-1])
        # rnn_states_critic = _cast(self.rnn_states_critic[:-1])
        rnn_states = self.rnn_states[:-1].transpose(1, 2, 0, 3, 4).reshape(-1, *self.rnn_states.shape[3:])
        rnn_states_critic = self.rnn_states_critic[:-1].transpose(1, 2, 0, 3, 4).reshape(-1, *self.rnn_states_critic.shape[3:])
        
        if self.available_actions is not None:
            available_actions = _cast(self.available_actions[:-1])

        for indices in sampler:

            if self._mixed_obs:
                share_obs_batch = defaultdict(list)
                obs_batch = defaultdict(list)
            else:
                share_obs_batch = []
                obs_batch = []

            rnn_states_batch = []
            rnn_states_critic_batch = []
            actions_batch = []
            available_actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            active_masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for index in indices:

                ind = index * data_chunk_length
                # size [T+1 N M Dim]-->[T N M Dim]-->[N,M,T,Dim]-->[N*M*T,Dim]-->[L,Dim]
                if self._mixed_obs:
                    for key in share_obs.keys():
                        share_obs_batch[key].append(share_obs[key][ind:ind+data_chunk_length])
                    for key in obs.keys():
                        obs_batch[key].append(obs[key][ind:ind+data_chunk_length])
                else:
                    share_obs_batch.append(share_obs[ind:ind+data_chunk_length])
                    obs_batch.append(obs[ind:ind+data_chunk_length])

                actions_batch.append(actions[ind:ind+data_chunk_length])
                if self.available_actions is not None:
                    available_actions_batch.append(available_actions[ind:ind+data_chunk_length])
                value_preds_batch.append(value_preds[ind:ind+data_chunk_length])
                return_batch.append(returns[ind:ind+data_chunk_length])
                masks_batch.append(masks[ind:ind+data_chunk_length])
                active_masks_batch.append(active_masks[ind:ind+data_chunk_length])
                old_action_log_probs_batch.append(action_log_probs[ind:ind+data_chunk_length])
                adv_targ.append(advantages[ind:ind+data_chunk_length])
                # size [T+1 N M Dim]-->[T N M Dim]-->[N M T Dim]-->[N*M*T,Dim]-->[1,Dim]
                rnn_states_batch.append(rnn_states[ind])
                rnn_states_critic_batch.append(rnn_states_critic[ind])
            
            L, N = data_chunk_length, mini_batch_size

            # These are all from_numpys of size (L, N, Dim) 
            if self._mixed_obs:
                for key in share_obs_batch.keys():  
                    share_obs_batch[key] = np.stack(share_obs_batch[key], axis=1)
                for key in obs_batch.keys():  
                    obs_batch[key] = np.stack(obs_batch[key], axis=1)
            else:        
                share_obs_batch = np.stack(share_obs_batch, axis=1)
                obs_batch = np.stack(obs_batch, axis=1)

            actions_batch = np.stack(actions_batch, axis=1)
            if self.available_actions is not None:
                available_actions_batch = np.stack(available_actions_batch, axis=1)
            value_preds_batch = np.stack(value_preds_batch, axis=1)
            return_batch = np.stack(return_batch, axis=1)
            masks_batch = np.stack(masks_batch, axis=1)
            active_masks_batch = np.stack(active_masks_batch, axis=1)
            old_action_log_probs_batch = np.stack(old_action_log_probs_batch, axis=1)
            adv_targ = np.stack(adv_targ, axis=1)

            # States is just a (N, -1) from_numpy
            rnn_states_batch = np.stack(rnn_states_batch).reshape(N, *self.rnn_states.shape[3:])
            rnn_states_critic_batch = np.stack(rnn_states_critic_batch).reshape(N, *self.rnn_states_critic.shape[3:])
            
            # Flatten the (L, N, ...) from_numpys to (L * N, ...)
            if self._mixed_obs:
                for key in share_obs_batch.keys(): 
                    share_obs_batch[key] = _flatten(L, N, share_obs_batch[key])
                for key in obs_batch.keys(): 
                    obs_batch[key] = _flatten(L, N, obs_batch[key])
            else:
                share_obs_batch = _flatten(L, N, share_obs_batch)
                obs_batch = _flatten(L, N, obs_batch)
            actions_batch = _flatten(L, N, actions_batch)
            if self.available_actions is not None:
                available_actions_batch = _flatten(L, N, available_actions_batch)
            else:
                available_actions_batch = None
            value_preds_batch = _flatten(L, N, value_preds_batch)
            return_batch = _flatten(L, N, return_batch)
            masks_batch = _flatten(L, N, masks_batch)
            active_masks_batch = _flatten(L, N, active_masks_batch)
            old_action_log_probs_batch = _flatten(L, N, old_action_log_probs_batch)
            adv_targ = _flatten(L, N, adv_targ)

            yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, adv_targ, available_actions_batch

    def reset_buffer(self):
        self.step = 0
        self.update_step = np.zeros((self.n_rollout_threads, self.num_agents, 1), dtype=np.int32)
        if self._mixed_obs:
            for key in self.share_obs.keys():
                self.share_obs[key].fill(0)
            for key in self.obs.keys():
                self.obs[key].fill(0)
        else:
            self.share_obs.fill(0)
            self.obs.fill(0)
        self.rnn_states.fill(0)
        self.rnn_states_critic.fill(0)
        self.value_preds.fill(0)
        self.returns.fill(0)
        if self.available_actions is not None:
            self.available_actions.fill(1)
        self.actions.fill(0)
        self.action_log_probs.fill(0)
        self.rewards.fill(0)
        self.masks.fill(1)
        if self.bad_masks is not None:
            self.bad_masks.fill(1)
        if self.active_masks is not None:
            self.active_masks.fill(1)