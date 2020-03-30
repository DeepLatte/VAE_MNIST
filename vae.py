import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import normal
from module import init_linear

'''
variational auto-encoder
'''

class VAE(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(VAE, self).__init__()
        self.vae_enc = ENC(in_dim=in_dim, 
                           hidden_dim=hidden_dim,
                           out_dim = out_dim)
        self.vae_dec = DEC(in_dim=out_dim, 
                           hidden_dim=hidden_dim,
                           out_dim = in_dim)

    def forward(self, input):
        mu, sigma = self.vae_enc(input)
        output = self.vae_dec(mu, sigma)

        return output, mu, sigma

class ENC(nn.Module):
    # return mu and sigma together
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(ENC, self).__init__()
        self.module_dict = nn.ModuleDict()
        self.module_dict.add_module('linear1', init_linear(in_dim, hidden_dim*2, True))
        self.module_dict.add_module('elu1', nn.ELU())
        self.module_dict.add_module('linear2', init_linear(hidden_dim*2, hidden_dim, True))
        self.module_dict.add_module('elu2', nn.ELU())
        self.module_dict.add_module('linear_out', init_linear(hidden_dim, out_dim*2, True))

    def forward(self, input):
        # should be (B, C, H, W) ---flatten----> (B, C, HXW)
        x = input
        for layer in self.module_dict:
            x = self.module_dict[layer](x)

        mu, sigma =  x.chunk(2, 1)
        sigma = 1e-6 + F.softplus(sigma) # sigma should be positive. 
        return mu, sigma 

class DEC(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(DEC, self).__init__()
        self.out_dim = out_dim
        self.module_dict = nn.ModuleDict()
        self.module_dict.add_module('linear1', init_linear(in_dim, hidden_dim, True))
        self.module_dict.add_module('elu1', nn.ELU())
        self.module_dict.add_module('linear2', init_linear(hidden_dim, hidden_dim*2, True))
        self.module_dict.add_module('elu2', nn.ELU())
        self.module_dict.add_module('linear_out', init_linear(hidden_dim*2, out_dim, True))
    
    def forward(self, mu, sigma):
        '''
        input
            mu : (B, in_dim)
            sigma : (B, in_dim)
        '''

        M = normal.Normal(0, 1)
        if str(mu.device) != "cpu":
            x = mu + sigma * M.sample(mu.size()).cuda()

        for layer in self.module_dict:
            x = self.module_dict[layer](x)
        output = torch.clamp(x, 1e-6, 1 - 1e-6)
        return output
        
class VAELoss:
    def __init__(self):
        try :
            self.reconst_error = nn.BCELoss().cuda() # BCE only support the input in a range -1 < x < 1
        except :
            self.reconst_error = nn.BCELoss()

    def __call__(self, x, y, mu, sigma):
        mrg_likelihood = self.reconst_error(x, y)
        kl_div = self.KLD(mu, sigma)
        
        return -(mrg_likelihood - kl_div)

    @staticmethod
    def KLD(mu, sigma):
        res = (mu * mu) + (sigma * sigma) - torch.log(1e-8 + (sigma * sigma)) - 1
        res = torch.mean(0.5 * torch.sum(res, dim=1))

        return res