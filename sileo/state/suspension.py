import torch

class Suspension(object):

    def __init__(self, device=torch.device('cuda'), mode='stereo'):
        self.device = device
        self.mode = mode

    def hang(self, tensor):
        '''
        Performs either unit normalisation, no projection at all or a stereographic projection
        '''
        if self.mode == 'norm':
            n= torch.linalg.norm(tensor)
            return torch.cat((tensor/n, torch.tensor([n]).to(self.device)))
        elif self.mode == 'id':
            return tensor
        else:
            khi = torch.linalg.norm(tensor)**2
            lamb = 2/(khi+1)
            return torch.cat((lamb*tensor, torch.tensor([1-lamb]).to(self.device)))

    def unhang(self, x):
        '''
        Inverse function of hang
        '''
        if self.mode == 'norm':
            return x[:-1]*x[-1]
        elif self.mode == 'id':
            return x
        else:
            lamb = 1 - x[-1]
            return x[:-1]/lamb