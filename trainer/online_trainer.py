#CONFORME

from time import time

import torch
import numpy as np
from tensordict.tensordict import TensorDict
from ville.action import defs
from ville.state.suspension import Suspension


from trainer.base import Trainer


class OnlineTrainer(Trainer):
	"""Trainer class for single-task online TD-MPC2 training."""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._step = 0
		self._ep_idx = 0
		self._start_time = time()
		self.match = self.cfg.match_actions
		self.match = False
		self.s = Suspension()


	def common_metrics(self):
		"""Return a dictionary of current metrics."""
		return dict(
			step=self._step,
			episode=self._ep_idx,
			total_time=time() - self._start_time,
		)

	def eval(self):
		"""Evaluate a TD-MPC2 agent."""
		ep_rewards = []
		split = int(np.floor(self.cfg.num_envs * self.cfg.split_frac))

		for i in range(self.cfg.eval_episodes // self.cfg.num_envs):
			obs, done, ep_reward, t = self.env.reset(), torch.tensor(False), 0, 0
			if self.cfg.save_video:
				self.logger.video.init(self.env, enabled=(i==0))
			while not done.any():
				action = self.agent.act(obs, t0=t==0, eval_mode=True)
				
				if isinstance(action[0], defs.Action): #action is a list of actions then
					if self.cfg.split_frac<1:
						act = torch.cat((torch.stack([a.embedding for a in action[:split]]), torch.stack(action[split:]).cpu()))
					else:
						act = torch.stack([a.embedding for a in action[:split]])
					obs = obs.to(self.agent.device, non_blocking=True)
					obs, reward, done, info = self.env.step(act)
					action = act				
				else:
					obs, reward, done, info = self.env.step(action)

				ep_reward += reward
				t += 1
				if self.cfg.save_video:
					self.logger.video.record(self.env)
			assert done.all(), 'Vectorized environments must reset all environments at once.'
			ep_rewards.append(ep_reward)
			if self.cfg.save_video:
				self.logger.video.save(self._step)
		return dict(
			episode_reward=torch.cat(ep_rewards).mean(),
			episode_success=info['success'].mean(),
		)

	def to_td(self, obs, action=None, reward=None):
		"""Creates a TensorDict for a new episode."""
		if isinstance(obs, dict):
			obs = TensorDict(obs, batch_size=(), device='cpu')
		else:
			obs = obs.unsqueeze(0).cpu()
		if action is None:
			action = torch.full_like(self.env.rand_act(), float('nan'))
		if reward is None:
			reward = torch.tensor(float('nan')).repeat(self.cfg.num_envs)
		td = TensorDict(dict(
			obs=obs,
			action=action.unsqueeze(0),
			reward=reward.unsqueeze(0),
		), batch_size=(1, self.cfg.num_envs,))
		return td

	def train(self):
		"""Train a TD-MPC2 agent."""
		train_metrics, done, eval_next = {}, torch.tensor(True), True
		hang = self.s.hang
		unhang = self.s.unhang
		split = int(np.floor(self.cfg.num_envs * self.cfg.split_frac))


		while self._step <= self.cfg.steps:

			# Evaluate agent periodically
			if self._step % self.cfg.eval_freq == 0:
				eval_next = True

			# Reset environment
			if done.any():
				assert done.all(), 'Vectorized environments must reset all environments at once.'
				if eval_next:
					eval_metrics = self.eval()
					eval_metrics.update(self.common_metrics())
					self.logger.log(eval_metrics, 'eval')
					eval_next = False

				if self._step > 0:
					tds = torch.cat(self._tds)
					train_metrics.update(
						episode_reward=tds['reward'].nansum(0).mean(),
						episode_success=info['success'].nanmean(),
					)
					train_metrics.update(self.common_metrics())
					self.logger.log(train_metrics, 'train')
					self._ep_idx = self.buffer.add(tds)

				obs = self.env.reset()
				self._tds = [self.to_td(obs)]

			# Collect experience
			if self._step > self.cfg.seed_steps:
				action = self.agent.act(obs, t0=len(self._tds)==1)

				if isinstance(action[0], defs.Action): #action is a list of actions then
					if self.cfg.split_frac<1:
						act = torch.cat((torch.stack([a.embedding for a in action[:split]]), torch.stack(action[split:]).cpu()))
					else:
						act = torch.stack([a.embedding for a in action[:split]])
					obs = obs.to(self.agent.device, non_blocking=True)
					z = obs
					obs, reward, done, info = self.env.step(act)
					y = obs.to(self.agent.device, non_blocking=True)
					for i in range(split):
						action[i].update_buffer(torch.stack((hang(z[i]), hang(y[i]))))
						for u in self.agent.completed_actions[i]:
							n = len(u.actions)
							u.update_buffer(torch.stack([hang(self.agent.buffer[-n][i]), hang(y[i])]))
					action = act				
				else:
					obs, reward, done, info = self.env.step(action)

			else:
				action = self.env.rand_act()
				obs, reward, done, info = self.env.step(action)
			self._tds.append(self.to_td(obs, action, reward))

			# Update agent
			if self._step >= self.cfg.seed_steps:
				if self._step == self.cfg.seed_steps:
					num_updates = int(self.cfg.seed_steps / self.cfg.steps_per_update)
					print('Pretraining agent on seed data...')
				else:
					num_updates = max(1, int(self.cfg.num_envs / self.cfg.steps_per_update))
				agent_thought = self.agent.thought
				for u in range(num_updates):
					if agent_thought and u == np.floor(num_updates * self.cfg.split_frac):
						self.agent.thought = False
					_train_metrics = self.agent.update(self.buffer)
				self.agent.thought = agent_thought
				train_metrics.update(_train_metrics)
    
				if self.think_every and self._step>self.pretrain_frac*self.cfg.steps and self.agent.thought:
					if self._step % self.think_every == 0:
						self.agent.think()
						if self.match and self.agent.fp is not None:
							self.agent.match_actions()

				elif self._step == int(self.pretrain_frac*self.cfg.steps):
					self.agent.think()
					if self.match:
						self.agent.match_actions()
				
				
				# if  self.match and self.agent.thought:
				# 	self.agent.match_actions(self.cfg.pretrained_model_path)
				# 	self.match = False

				
			self._step += self.cfg.num_envs

		self.logger.finish(self.agent)