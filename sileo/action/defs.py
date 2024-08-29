import torch
import numpy
import copy


class Action():
    
    def __init__(self, cfg, transitions, centroid):
        self.device = torch.device('cuda')
        self.ndim = transitions.shape[-1] #Dimension of the space upon which the actions act
        self.buffer_min_length = max(cfg.action_dim, cfg.latent_dim) #Action buffer used for rotation updates
        self.buffer_max_length = cfg.buffer_leeway * self.buffer_min_length
        self.buffer = transitions[-self.buffer_max_length:]
        self.rotation = torch.eye(self.ndim)
        self.embedding = torch.tensor(centroid) #embedding of the cluster's center; could be made to update alongside the rotation although unstable
        self.occurrences = len(transitions)
        self.rotate_step = cfg.rotate_step #Frequency at which an action should be updated
        
        self.update_rotation()

    
    def update_buffer(self, observation): #observation = tensor of shape (2, self.ndim) with initial and final state
        '''
        This method updates the action's buffer whenever it is chosen by the agent during online training. 
        It also regularly triggers an update of the action's rotation matrix
        '''
        n_entries = len(self.buffer)
        if n_entries < self.buffer_max_length: # Discard old observations
            self.buffer = torch.cat((self.buffer, observation.unsqueeze(0)))
        else:
            shifted_buffer = self.buffer.roll(-1, 0)
            shifted_buffer[-1] = observation
            self.buffer = shifted_buffer
        self.occurrences +=1
        if self.occurrences // self.rotate_step ==0: # Update trigger
            self.update_rotation()
    
    def calculate_B(self, buffer): # Buffer input is assumed to be a tensor of dimensions 1 x 2 x state_dim, i.e a list of 2-uples (initial state, final state)
        p = buffer.swapdims(0, 1)
        '''
        Calculates the "weighted" observation matrix B to be used in SVD as described in our paper 
        '''
        return torch.sum(torch.matmul(p[1].unsqueeze(1).swapdims(1, 2), p[0].unsqueeze(1)), dim = 0)
    
    def calculate_sigma(self, U, Vh):
        '''
        Calculates the intermediary matrix to substitute the diagonal singular matrix with in B's
        SVD to yield the solution following de Ruiter and Forbe's method.
        '''
        sigma = torch.eye(self.buffer.shape[-1])
        sigma[-1][-1] = torch.det(U)*torch.det(Vh)
        return sigma.to(self.device)
    
    def update_rotation(self):
        '''
        Updates the stored rotation with the most recent experimental observations
        '''
        window = self.buffer.to(self.device)
        B = self.calculate_B(window)
        U, _, Vh = torch.linalg.svd(B)
        sigma = self.calculate_sigma(U, Vh)

        self.rotation = torch.matmul(torch.matmul(U, sigma), Vh)

    def flatten(self):
        '''
        Auxiliary method used to stack completed actions when a composite action is called
        '''
        return [self]
    
    def distance(self, action): #Geodesic distance calculation between self and another instance of Action
        values = torch.angle(tuple(torch.linalg.eig(torch.transpose(self.rotation, 0, 1) @ action.rotation))[0])
        return torch.sqrt(torch.sum(values*values))
        

class Composite_Action(Action):

    def __init__(self, cfg, transitions, first_action, second_action):
        
        assert(isinstance(first_action, Action) and isinstance(second_action, Action))
        
        super().__init__(cfg, transitions, first_action.embedding)
        self.rotation = torch.matmul(second_action.rotation, first_action.rotation)
        self.parents = [first_action, second_action] #Unused in practice, can be used to reconstruct an inheritance graph
        
        # Creates the sequence of actions the composite action represents
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
        '''
        Returns the product of all the rotations of the primitive actions that constitute this composite action
        '''
        if len(actions) > 1:
            return torch.matmul(actions[0].rotation, self.assemble(actions[1:]))
        else:
            return torch.eye(self.ndim)
        
    def flatten(self):
        '''
        Rebuilds a list of primitive actions for the agent to execute when it decides to use this composite action
        '''
        first_action, second_action = self.parents
        return  first_action.flatten() + second_action.flatten() + [self]

    def yield_discrepancy(self):
        '''
        Yields the discrepancy between empirical effet of the sequence of actions (contained in self.rotation) and
        the expected effect of sequentially choosing each of its component primitives instead in planning
        '''
        return torch.linalg.matrix_norm(self.assemble(self.actions) - self.rotation, ord="2") # Operator norm


        