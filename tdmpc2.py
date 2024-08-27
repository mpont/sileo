import numpy as np
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import calinski_harabasz_score
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.neighbors import KDTree

from common import math
from common.scale import RunningScale
from common.world_model import WorldModel
from collections import deque

from ville.action import defs
from ville.state.suspension import Suspension

class SILEO_TDMPC2:
	"""
	Adapted TD-MPC2 agent. Implements training + inference.
	Can be used for single-task experiments,
	and supports state observations.
	"""

	def __init__(self, cfg, actions=None, model = None, fp=None):
		self.cfg = cfg
		self.device = torch.device('cuda')
		self.model = WorldModel(cfg).to(self.device)
		self.optim = torch.optim.Adam([
			{'params': self.model._encoder.parameters(), 'lr': self.cfg.lr*self.cfg.enc_lr_scale},
			{'params': self.model._dynamics.parameters()},
			{'params': self.model._reward.parameters()},
			{'params': self.model._Qs.parameters()},
			{'params': self.model._task_emb.parameters() if self.cfg.multitask else []}
		], lr=self.cfg.lr)
		self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=self.cfg.lr, eps=1e-5)
		self.model.eval()
		self.scale = RunningScale(cfg)
		self.cfg.iterations += 2*int(cfg.action_dim >= 20) # Heuristic for large action spaces
		self.discount = torch.tensor(
			[self._get_discount(ep_len) for ep_len in cfg.episode_lengths], device='cuda'
		) if self.cfg.multitask else self._get_discount(cfg.episode_length)
		
		self.buffer_max_length = cfg.buffer_max_length #Buffer to use while pruning
		print(self.buffer_max_length)
		self.n_entries = 0
		self.thought = False #Indicates if pre-training has ended
		self.reshuffle= False #Used if instead of refining an action set a new action set should be used after each pruning
		self.buffer = []
		self.action_list = [] #Stores all available Actions during training
		self.action_embeddings = [] #KDTree used in practice to retrieve nearest action given an embedding
		self.stacked_actions = [deque() for _ in range(self.cfg.num_envs)] #Used to unstack a sequence of primitives once a composite action was chosen
		self.completed_actions = [] #Lists all actions to update once a primitive has been executed
		self.first_prune = cfg.first_prune #Checks if composite actions should be introduced alongside the first primitives
		self.pruning_factor = cfg.pruning_factor # Number of medians to use as threshold
		self.k_means_attempts = cfg.k_means_attempts #Used in clustering
		self.s = Suspension(self.device) #Instantiates the type of projections to use
		self.split = int(np.floor(self.cfg.num_envs * self.cfg.split_frac)) #Split between discrete and continuous environments
		
		#Used to transfer knowledge from an old task
		self.old_actions = actions 
		self.old_model = model
		self.fp = fp


	def _get_discount(self, episode_length):
		"""
		Returns discount factor for a given episode length.
		Simple heuristic that scales discount linearly with episode length.
		Default values should work well for most tasks, but can be changed as needed.

		Args:
			episode_length (int): Length of the episode. Assumes episodes are of fixed length.

		Returns:
			float: Discount factor for the task.
		"""
		frac = episode_length/self.cfg.discount_denom
		return min(max((frac-1)/(frac), self.cfg.discount_min), self.cfg.discount_max)

	def save(self, fp):
		"""
		Save state dict and action list (if available) of the agent to filepath.
		
		Args:
			fp (str): Filepath to save state dict to.
		"""
		if self.action_list:
			torch.save({"model": self.model.state_dict()}, fp)
			a = np.array([np.concatenate([np.array(u) for u in _a.rotation.cpu()], axis = None) for _a in self.action_list])
			print(len(a[0]))
			b = np.array(torch.stack([_a.embedding for _a in self.action_list]).cpu())
			np.savetxt("actions.csv", a, delimiter=",")
			np.savetxt("embeddings.csv", b, delimiter = ",")
		else:
			torch.save({"model": self.model.state_dict()}, fp)

	def load(self, fp, load_model = True, load_actions = False):
		"""
		Load a saved state dict from filepath (or dictionary) into current agent.
		
		Args:
			fp (str or dict): Filepath or state dict to load.
		"""
		state_dict = fp if isinstance(fp, dict) else torch.load(fp)
		
		if load_model:
			self.model.load_state_dict(state_dict["model"])

	def match_actions(self, task=None):
		'''
		Used to match actions used in a previous task with actions currently used.
		Computes distance between these tasks using mean geodesic distance between matched actions.
		'''
		
		second_action_list = self.old_actions
		
		distances = np.array([[a.distance(b) for b in second_action_list] for a in self.action_list])
		new_actions_preferences = np.array([np.argsort(distances[u]) for u in range(len(self.action_list))])
		old_actions_preferences = np.array([np.argsort(distances.T[u]) for u in range(len(second_action_list))])
		matches = [-1 for u in self.action_list]
		
		# Gale-Shapley
		try:
			while matches.index(-1) > -1:
				u = matches.index(-1)
				preferences = new_actions_preferences[u]
				for j in preferences:
					try:
						v = matches.index(j)
						if old_actions_preferences[j].index(v) > old_actions_preferences[j].index(u):
							matches[v], matches[u] = matches[u], matches[v]
							
					except:
						matches[u] = j
						print("matched")
						
		except:
			matching = [[self.action_list[i], second_action_list[matches[i]]] for i in range(len(self.action_list))]
			distances = [u[0].distance(u[1]) for u in matching]
			print(f"Average distance:{sum(distances)/len(matching)}")
			print(f"Max distance:{max(distances)/len(matching)}")
			print(f"Min distance:{min(distances)}")

			self.action_list = [second_action_list[matches[i]] for i in range(len(self.action_list))]



		
	@torch.no_grad()
	def store_observation(self, observation):
		'''
		Stores observation passed to the agent in the buffer and ensures
		the buffer does not exceed a certain length
		'''
		if isinstance(self.buffer, torch.Tensor):
			self.buffer = self.buffer.roll(-1)
			self.buffer[-1] = observation
		else:
			if len(self.buffer) < self.buffer_max_length:
				self.buffer.append(observation)
			else:
				self.buffer[:] = self.buffer[-self.buffer_max_length//2:]
				self.buffer.append(observation)

		

	@torch.no_grad()
	def cluster(self, actions):
	# Find optimal number centroids for the actions seen through the buffer
		score = 0
		argmax = 0
		for i in range(8, self.cfg.max_primitives, 2):
			assignment = KMeans(n_clusters=i, n_init=3).fit(actions).labels_
			sc = calinski_harabasz_score(actions, assignment)
			if sc>score:
				score = sc
				argmax = i
		return KMedoids(n_clusters = 2*argmax).fit(actions)

	@torch.no_grad()
	def think(self, eval_mode = False, task = None):
		'''
		Essentially the pruning algorithm described in the paper;
		Clusters actions in the buffer to find relevant new primitives
		Refines these primitives into larger sequences with composite actions
		'''
		hang = self.s.hang
		
		if not self.thought:
			print("Starting to reflect...")
			#Clustering of primitives
			if task is not None:
				task = torch.tensor([task], device=self.device)
			if len(self.buffer) > 2* self.cfg.buffer_max_length:
				self.buffer[:] = self.buffer[-self.buffer_max_length:]
			actions = tuple([self.act(obs, task, think= True, override = True) for obs in self.buffer])
			base = np.array(torch.cat(actions))
			kmeans = self.cluster(base)
			labels, centroids = kmeans.labels_, kmeans.cluster_centers_
			k = len(centroids)
			transitions = [torch.tensor([]) for i in range(k)]
			self.action_list[:] = []

			buffer = tuple(self.buffer)
			buffer = torch.cat(buffer)
			buffer = torch.stack([hang(obs) for obs in buffer])
						
			if len(buffer.size()) > 2:
				buffer = buffer.mean(axis= self.buffer.size().index(self.cfg.num_envs))
			
			#Builds actions from the cluster centers using the assigned actions' transitions as theirs
			for centroid in range(k):	
				assignments = [torch.stack((buffer[i], buffer[i+1])) for i, label in enumerate(labels)
				if (label==centroid and i%len(self.buffer)+1 != 0 and i<len(buffer) -1)]
				try:
					transitions[centroid] = torch.stack(assignments)
				except:
					transitions[centroid] = torch.eye(buffer.shape[-1])
			print(f"Number of primitive skills :{k}")
			for u in range(k):
				try:
					self.action_list.append(defs.Action(self.cfg, transitions[u], centroids[u]))
				except:
					pass

			#Build composite actions
			if self.first_prune:
				occ = [a.occurrences for a in self.action_list]
				median = torch.median(torch.tensor(occ))
				mean = torch.mean(torch.tensor(occ), dtype = torch.float32)
				std = torch.std(torch.tensor(occ, dtype = torch.float32))

				while max(occ)>self.pruning_factor*median:
					#Find overused primitive and evaluate likelihood of possible continuations
					ind = occ.index(max(occ))
					action = self.action_list[ind]
					distribution = [0 for u in centroids]
					transitions = [[] for u in centroids]
					
					for centroid in range(k):
						assignments = [torch.stack((buffer[i], buffer[i+2])) for i, label in enumerate(labels)
						if (label==ind and i%len(self.buffer)+1 != 0 and i<len(buffer) -2 and labels[i+1] == centroid)]
						if len(assignments)>0:
							transitions[centroid] = assignments
					distribution = [len(u) for u in transitions]
	
					reassigned = 0
					# Build composite actions from most likely continuations
					for j in range(k):
						try:
							if distribution[j]>min(median*self.pruning_factor, mean+2*std):
								self.action_list.append(defs.Composite_Action(self.cfg, torch.stack(transitions[j]), action, self.action_list[j]))
								self.action_list[ind].occurrences -= len(transitions[j])
								occ.append(len(transitions[j]))
								reassigned += len(transitions[j])
						except:
							pass

					if reassigned == 0: # Abandon
						occ[ind] = 0					

			
			self.action_embeddings = KDTree(np.array(torch.stack([a.embedding for a in self.action_list]).cpu()), leaf_size = k//3)
			self.model.thought = True
			self.thought = True
			print("Finished thinking")
			print(f"Total number of actions {len(self.action_list)}")

		elif self.reshuffle:
			self.thought = False
			self.think(eval_mode = False, task = None)

		else:
			self.prune(eval_mode = eval_mode, task = task)

	@torch.no_grad()
	def find_action(self, state, embedding, task = None): # Note: state here is a SINGLE state, not a batch
	'''
	Finds an action given a proposed embedding a state by evaluating discounted rewards 
	after execution of all actions that "start" with the nearest primitive action
	'''
		distances, _ = self.action_embeddings.query(np.array(embedding.unsqueeze(0).cpu().nan_to_num(0)))
		d = min(distances)
		candidates = [i for i, dist in enumerate(distances) if dist == d]
		valuations = []

		if len(candidates) >1:
			for index in candidates:
				action = self.action_list[index]
				valuations.append(self._estimate_value(z=state, actions= action, task = task).nan_to_num_(0))
		else:
			return candidates[0]
		
		return candidates[valuations.index(max(valuations))] # Returns index in self.action_list of the action to use

	@torch.no_grad()
	def prune(self, eval_mode = False, task = None):
		'''
		Use for repeated pruning in downstream learning, essentially identical to think
		'''
		assert(self.thought)
		#Overhead
		unhang = self.s.unhang
		hang = self.s.hang
		self.thought = False
		if len(self.buffer) > 2* self.cfg.buffer_max_length or len(self.buffer) > 20000:
			self.buffer = self.buffer[-self.buffer_max_length:]
		actions = tuple([self.act(obs, task, think= True, override = True) for obs in self.buffer])
		base = np.array(torch.cat(actions))
		self.thought = True

		#Cluster and find relevant new primitives
		kmeans = self.cluster(base)
		labels, centroids = kmeans.labels_, kmeans.cluster_centers_
		k = len(centroids)
		distances = [self.action_embeddings.query([embedding], k=1)[0] for embedding in centroids]
		cutoff, std = np.median(np.array(distances, dtype = np.float32)), np.std(np.array(distances, dtype = np.float32))
		indices = [i for i, distance in enumerate(distances) if distance > cutoff + std]

		transitions = [torch.tensor([]) for i in centroids]

		buffer = tuple(self.buffer)
		buffer = torch.cat(buffer)
		buffer = torch.stack([hang(obs) for obs in buffer])
					
		if len(buffer.size()) > 2:
			buffer = buffer.mean(axis= self.buffer.size().index(self.cfg.num_envs))
		print(f"buffer shape: {buffer.shape}")
		for centroid in indices:	
			assignments = [torch.stack((buffer[i], buffer[i+1])) for i, label in enumerate(labels)
			if (label==centroid and i%len(self.buffer)+1 != 0 and i<len(buffer) -1)]
			try:
				transitions[centroid] = torch.stack(assignments)
			except:
				transitions[centroid] = torch.eye(buffer.shape[-1])

		print(f"Number of new primitive skills :{len(indices)}")
		for centroid in indices:
			self.action_list.append(defs.Action(self.cfg, transitions[centroid], centroids[centroid]))


		#Build new composite actions
		occ = [a.occurrences for a in self.action_list]
		median = torch.median(torch.tensor(occ))
		mean = torch.mean(torch.tensor(occ, dtype = torch.float32))
		std = torch.std(torch.tensor(occ, dtype = torch.float32))
		while max(occ) > min(self.pruning_factor*median, median + 2*std):
			ind = occ.index(max(occ))
			root_action = self.action_list[ind]
			distribution = [0 for u in self.action_list]
			buffer = root_action.buffer
			transitions = [torch.tensor([]).to(self.device) for i in range(len(self.action_list))]

			# Evaluate the best continuation for the most used action
			self.thought = False
			for u in buffer:
				v = unhang(u[1]).to(self.device, non_blocking=True)
				u = unhang(u[0]).to(self.device, non_blocking=True)
				self.thought = False
				embedding = self.act(v, task, think= True, override = True)
				action2 = self.find_action(v, embedding, task)
				distribution[action2] += 1
				if len(transitions[action2])==0:
					transitions[action2] = torch.cat((transitions[action2], torch.stack((hang(u), torch.matmul(self.action_list[action2].rotation, hang(v))))))
					n = True
				elif n:
					n = False
					transitions[action2] = torch.stack((transitions[action2], torch.stack((hang(u), torch.matmul(self.action_list[action2].rotation, hang(v))))))
				else:
					transitions[action2] = torch.cat((transitions[action2], torch.stack((hang(u), torch.matmul(self.action_list[action2].rotation, hang(v)))).unsqueeze(0)))
			self.thought = True
			
			#Construct the new composite actions
			for j in range(len(distribution)):
				if distribution[j]>median/self.pruning_factor:
					try:
						self.action_list.append(defs.Composite_Action(self.cfg, transitions[j], root_action, self.action_list[j]))
						root_action.occurrences -= len(transitions[j])
					except:
						pass

			occ[ind] = 0 # Avoid looping on same action
		self.action_embeddings = KDTree(np.array(torch.stack([a.embedding for a in self.action_list])))	
		print(f"Total number of actions {len(self.action_list)}")
	

	@torch.no_grad()
	def act(self, obs, t0=False, eval_mode=False, task=None, think = False, override = False):
		"""
		Select an action by planning in the latent space of the world model.
		
		Args:
		obs (torch.Tensor): Observation from the environments. Shape (num_envs, observation_dim)
		t0 (bool): Whether this is the first observation in the episode.
		eval_mode (bool): Whether to use the mean of the action distribution.
		task (int): Task index (only used for multi-task experiments).
	
		Returns:
			torch.Tensor: Action to take in the environment.
		"""
		if len(obs.shape)==1:
			envs = 1
		else:
			envs = obs.shape[0]
		self.completed_actions[:] = [[] for _ in range(envs)]
		split = self.split


		obs = obs.to(self.device, non_blocking=True)
		if not think: #If not planning/reflecting on imagined states
			self.store_observation(obs)

		if task is not None:
			task = torch.tensor([task], device=self.device)
		z = self.model.encode(obs, task)
		
		out = [[] for _ in range(envs)]
		#Check if composite actions are being executed at current timestep and yield the next primitive
		for i in range(envs):
			if self.stacked_actions[i]:				
				out[i] = self.stacked_actions[i].popleft()
				while 0 < len(self.stacked_actions[i]) and isinstance(self.stacked_actions[i][0], defs.Composite_Action):
					self.completed_actions[i].append(self.stacked_actions[i].popleft())

		#Plan for the next action
		if self.cfg.mpc and not override:
			if not self.thought:
				a = self.plan(z, t0=t0, eval_mode=eval_mode, task=task)
			else:
				a = self.plan(obs, t0=t0, eval_mode=eval_mode, task=task)
		else:
			a = self.model.pi(z, task)[int(not eval_mode)]
		
		#Recast policy actions to the nearest primitives or useful composite actions and start their execution
		if self.thought:
			for i in range(split):
				if not out[i]:
					action = self.action_list[self.find_action(z[i], a[i], task)]
					self.stacked_actions[i].extend(action.flatten())
					out[i] = self.stacked_actions[i].popleft()
			#Used for continuous environments that continue working alongside our framework 
			if split<envs:
				out[split:] = list(a[split:])
			return out
			
		else:
			return a.cpu()

	@torch.no_grad()
	def _estimate_value(self, z, actions, task):
		"""Estimate value of a trajectory starting at latent state z and executing given actions."""
		G, discount = 0, 1
		t = 0
		while t < self.cfg.horizon:
			# Evaluate rewards taking into account uncertainty for composite actions
			if isinstance(actions, defs.Composite_Action):
				for a in actions.actions:
					reward = math.two_hot_inv(self.model.reward(z, a.embedding.to(self.device), task), self.cfg)
					z = self.model.next(z, a, task)
					G += discount * reward
					discount *= self.discount[torch.tensor(task)] if self.cfg.multitask else self.discount
				t+= self.cfg.horizon # Actions already had their planning value estimation, no need to repeat in-depth estimation
			elif isinstance(actions, defs.Action):
				a = actions
				reward = math.two_hot_inv(self.model.reward(z, a.embedding.to(self.device), task), self.cfg)
				z= self.model.next(z, actions, task)
				G += discount * reward
				discount *= self.discount[torch.tensor(task)] if self.cfg.multitask else self.discount
				t+= self.cfg.horizon # Actions already had their separate value estimation, no need to repeat in-depth estimation
			else:
				reward = math.two_hot_inv(self.model.reward(z, actions[:, t].to(self.device), task), self.cfg)
				z = self.model.next(z, actions[:, t], task)
				G += discount * reward
				discount *= self.discount[torch.tensor(task)] if self.cfg.multitask else self.discount
			t+=1

		return G + discount * self.model.Q(z, self.model.pi(z, task)[1], task, return_type='avg')

	@torch.no_grad()
	def plan(self, z, t0=False, eval_mode=False, task=None):
		"""
		Plan a sequence of actions using the learned world model.
		
		Args:
			z (torch.Tensor): Latent state from which to plan. Shape is (num_envs, latent_dim)
			t0 (bool): Whether this is the first observation in the episode.
			eval_mode (bool): Whether to use the mean of the action distribution.
			task (Torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			torch.Tensor: Action to take in the environment.
		"""		
		# Sample policy trajectories
		if self.thought:
			obs = z
			z = self.model.encode(obs, task)
		split = self.split


		if self.cfg.num_pi_trajs > 0:
			pi_actions = torch.empty(self.cfg.num_envs, self.cfg.horizon, self.cfg.num_pi_trajs, self.cfg.action_dim, device=self.device)
			_z = z.unsqueeze(1).repeat(1, self.cfg.num_pi_trajs, 1) # Shape (num_envs, num_pi_trajs, latent_dim)
			
			#Recast proposed policy actions as Actions and plan using them
			if self.thought:
				_obs = obs.unsqueeze(1).repeat(1, self.cfg.num_pi_trajs, 1)
			for t in range(self.cfg.horizon):
				pi_actions[:, t] = self.model.pi(_z, task)[1]
				if self.thought:	
					acts = np.array([[self.action_list[self.find_action(_z[env, traj], pi_actions[env, t, traj], task)].flatten()[0] 
			   				for traj in range(self.cfg.num_pi_trajs)] for env in range(split)], dtype = object)
					
					if t < self.cfg.horizon -1:
						_obs[:split] = self.model.next(_obs[:split], acts[:split], task)
						_z[:split] = self.model.encode(_obs[:split], task)
						if split<self.cfg.num_envs:
							_z[split:] = self.model.next(_z[split:], pi_actions[split:, t], task)
					
				
				else:
					acts = pi_actions[:, t]					
					if t < self.cfg.horizon -1:
						_z = self.model.next(_z, acts, task)
				
				# Return embeddings of the Actions chosen to be used for MPPI
				if self.thought:
					for i in range(split):
						for j in range(self.cfg.num_pi_trajs):
							pi_actions[i, t, j, :] = acts[i, j].embedding

		# Initialize state and parameters
		z = z.unsqueeze(1).repeat(1, self.cfg.num_samples, 1)
		mean = torch.zeros(self.cfg.num_envs, self.cfg.horizon, self.cfg.action_dim, device=self.device)
		std = self.cfg.max_std*torch.ones(self.cfg.num_envs, self.cfg.horizon, self.cfg.action_dim, device=self.device)
		if not t0:
			mean[:, :-1] = self._prev_mean[:, 1:]
		actions = torch.empty(self.cfg.num_envs, self.cfg.horizon, self.cfg.num_samples, self.cfg.action_dim, device=self.device)
		if self.cfg.num_pi_trajs > 0:
			actions[:, :, :self.cfg.num_pi_trajs] = pi_actions
	
		# Iterate MPPI
		for _ in range(self.cfg.iterations):

			# Sample actions
			actions[:, :, self.cfg.num_pi_trajs:] = (mean.unsqueeze(2) + std.unsqueeze(2) * \
				torch.randn(self.cfg.num_envs, self.cfg.horizon, self.cfg.num_samples-self.cfg.num_pi_trajs, self.cfg.action_dim, device=std.device)) \
				.clamp(-1, 1)
			if self.cfg.multitask:
				actions = actions * self.model._action_masks[task]

			# Compute elite actions
			value = self._estimate_value(z, actions, task).nan_to_num_(0)
			elite_idxs = torch.topk(value.squeeze(2), self.cfg.num_elites, dim=1).indices
			elite_value = torch.gather(value, 1, elite_idxs.unsqueeze(2))
			elite_actions = torch.gather(actions, 2, elite_idxs.unsqueeze(1).unsqueeze(3).expand(-1, self.cfg.horizon, -1, self.cfg.action_dim))

			# Update parameters
			max_value = elite_value.max(1)[0]
			score = torch.exp(self.cfg.temperature*(elite_value - max_value.unsqueeze(1)))
			score /= score.sum(1, keepdim=True)
			mean = torch.sum(score.unsqueeze(1) * elite_actions, dim=2) / (score.sum(1, keepdim=True) + 1e-9)
			std = torch.sqrt(torch.sum(score.unsqueeze(1) * (elite_actions - mean.unsqueeze(2)) ** 2, dim=2) / (score.sum(1, keepdim=True) + 1e-9)) \
				.clamp_(self.cfg.min_std, self.cfg.max_std)
			if self.cfg.multitask:
				mean = mean * self.model._action_masks[task]
				std = std * self.model._action_masks[task]

		# Select action sequence with probability `score`
		score = score.squeeze(1).squeeze(-1).cpu().numpy()
		actions = torch.stack([
				elite_actions[i, :, np.random.choice(np.arange(score.shape[1]), p=score[i])] \
			for i in range(score.shape[0])], dim=0)

		self._prev_mean = mean
		action, std = actions[:, 0], std[:, 0]
		if not eval_mode:
			action += std * torch.randn(self.cfg.action_dim, device=std.device)
		return action.clamp_(-1, 1)
		
	def update_pi(self, zs, task):
		"""
		Update policy using a sequence of latent states.
		
		Args:
			zs (torch.Tensor): Sequence of latent states.
			task (torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			float: Loss of the policy update.
		"""
		self.pi_optim.zero_grad(set_to_none=True)
		self.model.track_q_grad(False)
		_, pis, log_pis, _ = self.model.pi(zs, task)
		qs = self.model.Q(zs, pis, task, return_type='avg')
		self.scale.update(qs[0])
		qs = self.scale(qs)

		# Loss is a weighted sum of Q-values
		rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))
		pi_loss = ((self.cfg.entropy_coef * log_pis - qs).mean(dim=(1,2)) * rho).mean()
		pi_loss.backward()
		torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm)
		self.pi_optim.step()
		self.model.track_q_grad(True)

		return pi_loss.item()


	@torch.no_grad()
	def _td_target(self, next_z, reward, task):
		"""
		Compute the TD-target from a reward and the observation at the following time step.
		
		Args:
			next_z (torch.Tensor): Latent state at the following time step.
			reward (torch.Tensor): Reward at the current time step.
			task (torch.Tensor): Task index (only used for multi-task experiments).
		
		Returns:
			torch.Tensor: TD-target.
		"""
		pi = self.model.pi(next_z, task)[1]
		discount = self.discount[task].unsqueeze(-1) if self.cfg.multitask else self.discount
		return reward + discount * self.model.Q(next_z, pi, task, return_type='min', target=True)

	@torch.no_grad()
	def rollout(self, obs, action, task):
		'''
		Auxiliary function used to rollout intermediary states when executing a composite action
		'''
		states = []
		for act in action.actions:
			obs = self.model.next(obs, act, task)
			states.append(obs)
		return states
	
	def update(self, buffer):
		"""
		Main update function. Corresponds to one iteration of model learning.
		
		Args:
			buffer (common.buffer.Buffer): Replay buffer.
		
		Returns:
			dict: Dictionary of training statistics.
		"""
		obs, action, reward, task = buffer.sample()
		self.task = task
		hang = self.s.hang
	
		# Compute targets
		with torch.no_grad():
			next_z = self.model.encode(obs[1:], task)
			td_targets = self._td_target(next_z, reward, task)

		# Prepare for update
		self.optim.zero_grad(set_to_none=True)
		self.model.train()

		# Latent rollout
		zs = torch.empty(self.cfg.horizon+1, self.cfg.batch_size, self.cfg.latent_dim, device=self.device)
		z = self.model.encode(obs[0], task)
		zs[0] = z
		t = 0
		_obs = torch.zeros(obs.shape).to(self.device)
		_ob= torch.zeros(obs.shape[1:]).to(self.device)
		_obs[0] = obs[0]

		while t < self.cfg.horizon:
			if self.thought:
				with torch.no_grad():
					if t == 0:
						ts = np.zeros(self.cfg.batch_size, dtype = np.int16)
					# Sample actions for every element of the batch, plan and act using them before updating for rollout
					for b in range(self.cfg.batch_size):
						if ts[b] < self.cfg.horizon:
							act = self.action_list[self.find_action(z[b], action[ts[b], b], task)]
							if isinstance(act, defs.Composite_Action):
								if len(act.actions)+1>= self.cfg.horizon-ts[b]:
									act = act.actions[0]
									_ob[b] = self.model.next(_obs[ts[b]][b], act, task)
									z[b] = self.model.encode(_ob[b], task)
									_obs[ts[b]+1][b][:] = _ob[b]
									ts[b]+= 1
								else:
									_obs[ts[b]+1:ts[b]+len(act.actions)+1][b][:] = torch.tensor(np.array(self.rollout(_obs[ts[b]][b], act, task)))
									z[b] = self.model.encode(_obs[ts[b]+len(act.actions)][b], task)
									ts[b]+= len(act.actions) 
							else:
								_ob[b] = self.model.next(_obs[ts[b]][b], act, task)
								_obs[ts[b]+1][b][:] = _ob[b]
								z[b] = self.model.encode(_ob[b], task)
								ts[b]+= 1
							if ts[b]< self.cfg.horizon:
								act.update_buffer(torch.stack([hang(obs[ts[b]-1][b]), hang(obs[ts[b]][b])]))
					t = min(ts)
			else:
				#Baseline for pre-training
				z = self.model.next(z, action[t], task)
				zs[t+1] = z
				t+= 1

		if self.thought:
			#Back-and-forth through numpy to reroot the autograd computation graph and encode imagined observations
			zs = self.model.encode(torch.tensor(np.array(_obs.cpu())).to(self.device), task)
		
	

		# Predictions
		_zs = zs[:-1]
		qs = self.model.Q(_zs, action, task, return_type='all')
		reward_preds = self.model.reward(_zs, action, task)
		if self.thought:
			qs = torch.nan_to_num(qs, nan=1e-6)
			reward_preds = torch.nan_to_num(reward_preds, nan=1e-6)
		

		# Compute losses
		consistency_loss = sum([F.mse_loss(zs[t+1], next_z[t]) * self.cfg.rho**t for t in range(self.cfg.horizon-1)])
		
		if consistency_loss.isnan(): 
			consistency_loss= 1e-6

		reward_loss, value_loss = 0, 0

		for t in range(self.cfg.horizon):
			reward_loss += math.soft_ce(reward_preds[t], reward[t], self.cfg).mean() * self.cfg.rho**t
			for q in range(self.cfg.num_q):
				value_loss += math.soft_ce(qs[q][t], td_targets[t], self.cfg).mean() * self.cfg.rho**t
		consistency_loss *= (1/self.cfg.horizon)
		reward_loss *= (1/self.cfg.horizon)
		value_loss *= (1/(self.cfg.horizon * self.cfg.num_q))

		total_loss = (
			self.cfg.consistency_coef * consistency_loss +
			self.cfg.reward_coef * reward_loss +
			self.cfg.value_coef * value_loss
		)

		# Update model
		total_loss.backward()
		grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
		self.optim.step()

		# Update policy
		pi_loss = self.update_pi(zs.detach(), task)

		# Update target Q-functions
		self.model.soft_update_target_Q()

		# Return training statistics
		self.model.eval()
		return {
			"consistency_loss": float(consistency_loss.mean().item()),
			"reward_loss": float(reward_loss.mean().item()),
			"value_loss": float(value_loss.mean().item()),
			"pi_loss": pi_loss,
			"total_loss": float(total_loss.mean().item()),
			"grad_norm": float(grad_norm),
			"pi_scale": float(self.scale.value),
		}
