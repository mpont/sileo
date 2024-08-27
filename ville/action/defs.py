import torch
import numpy
import copy


class Action():
    
    def __init__(self, cfg, transitions, centroid):
        self.device = torch.device('cuda')
        self.ndim = transitions.shape[-1]
        self.buffer_min_length = max(cfg.action_dim, cfg.latent_dim) #cfg.manual_buffer if (cfg.min_length_override or cfg.manual_buffer > 0) else
        self.buffer_max_length = cfg.buffer_leeway * self.buffer_min_length
        self.buffer = transitions[-self.buffer_max_length:]
        self.rotation = torch.eye(self.ndim)
        self.embedding = torch.tensor(centroid)
        self.occurrences = len(transitions)
        self.rotate_step = cfg.rotate_step
        
        self.update_rotation()

    
    def update_buffer(self, observation): #observation = tenseur 2 x state_dim avec etat initial et final
        n_entries = len(self.buffer)
        if n_entries < self.buffer_max_length:
            self.buffer = torch.cat((self.buffer, observation.unsqueeze(0)))
        else:
            shifted_buffer = self.buffer.roll(-1, 0)
            shifted_buffer[-1] = observation
            self.buffer = shifted_buffer
        self.occurrences +=1
        if self.occurrences // self.rotate_step ==0:
            self.update_rotation()
    
    def calculate_B(self, buffer): # Buffer input is assumed to be a tensor of dim 1 x 2 x state_dim, I.E a list of 2-uples (initial state, final state)
        p = buffer.swapdims(0, 1)
        return torch.sum(torch.matmul(p[1].unsqueeze(1).swapdims(1, 2), p[0].unsqueeze(1)), dim = 0)
    
    def calculate_sigma(self, U, Vh):
        sigma = torch.eye(self.buffer.shape[-1])
        sigma[-1][-1] = torch.det(U)*torch.det(Vh)
        return sigma.to(self.device)
    
    def update_rotation(self):
        window = self.buffer.to(self.device)
        B = self.calculate_B(window)
        U, _, Vh = torch.linalg.svd(B)
        sigma = self.calculate_sigma(U, Vh)

        self.rotation = torch.matmul(torch.matmul(U, sigma), Vh)

    def flatten(self):
        return [self]
    
    def distance(self, action): #Geodesic distance
        values = torch.angle(torch.tensor(tuple(torch.linalg.eig(torch.transpose(self.rotation, 0, 1) @ action.rotation).cpu())))
        return torch.sqrt(torch.sum(torch.tensor(numpy.array([a*a for a in values]))))
        

class Composite_Action(Action):

    def __init__(self, cfg, transitions, first_action, second_action):
        
        assert(isinstance(first_action, Action) and isinstance(second_action, Action))
        
        super().__init__(cfg, transitions, first_action.embedding)
        self.rotation = torch.matmul(second_action.rotation, first_action.rotation)
        self.parents = [first_action, second_action]
        
        if isinstance(first_action, Composite_Action) and isinstance(second_action, Composite_Action):
            self.actions = first_action.actions + second_action.actions
        elif isinstance(second_action, Composite_Action):
                self.actions = [first_action] + second_action.actions
        elif isinstance(first_action, Composite_Action):
            self.actions = first_action.actions + [second_action]
        else:
            self.actions = [first_action]
            self.actions.append(second_action)

    def assemble(self, actions):
        if len(actions) > 1:
            return torch.matmul(actions[0].rotation, self.assemble(actions[1:]))
        else:
            return torch.eye(self.ndim)
        
    def flatten(self):
        first_action, second_action = self.parents
        return  first_action.flatten() + second_action.flatten() + [self]

    def yield_discrepancy(self):
        return torch.linalg.matrix_norm(self.assemble(self.actions) - self.rotation, ord="2") # Operator norm


        