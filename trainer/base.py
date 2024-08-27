class Trainer:
	"""Base trainer class for TD-MPC2."""

	def __init__(self, cfg, env, agent, buffer, logger):
		self.cfg = cfg
		self.env = env
		self.agent = agent
		self.buffer = buffer
		self.logger = logger

		self.pretrain_frac = cfg.pretrain_frac
		self.think_every = cfg.think_every # 0 means not to use this parameter

		if cfg.load_pretrained:
			if cfg.transfer_mode == 0: # Start with full model and parameters
				self.agent.load(cfg.pretrained_model, load_actions = True)
			elif cfg.transfer_mode == 1: # Only load action list
				self.agent.load(cfg.pretrained_model, load_model = False, load_actions = True)
		
		print('Architecture:', self.agent.model)
		print("Learnable parameters: {:,}".format(self.agent.model.total_params))

	def eval(self):
		"""Evaluate a TD-MPC2 agent."""
		raise NotImplementedError

	def train(self):
		"""Train a TD-MPC2 agent."""
		raise NotImplementedError
