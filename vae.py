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

    def infer_z(self, z):
        output = self.vae_dec.infer_z(z)
        return output

class ENC(nn.Module):
    # return mu and sigma together
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(ENC, self).__init__()
        self.module_dict = nn.ModuleDict()
        self.module_dict.add_module('linear1', init_linear(in_dim, hidden_dim*2, True))
        self.module_dict.add_module('ReLU1', nn.ReLU())
        self.module_dict.add_module('linear2', init_linear(hidden_dim*2, hidden_dim, True))
        self.module_dict.add_module('ReLU2', nn.ReLU())
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
        self.module_dict.add_module('ReLU1', nn.ReLU())
        self.module_dict.add_module('linear2', init_linear(hidden_dim, hidden_dim*2, True))
        self.module_dict.add_module('ReLU2', nn.ReLU())
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

        # x = mu + sigma * torch.rand_like(mu)
        for layer in self.module_dict:
            x = self.module_dict[layer](x)
        output = torch.clamp(x, 1e-6, 1 - 1e-6)
        return output

    def infer_z(self, z):
        for layer in self.module_dict:
            z = self.module_dict[layer](z)
            
        output = torch.clamp(z, 1e-6, 1 - 1e-6)
        return output

class VAELoss:
    def __init__(self):
        try :
            self.reconst_error = nn.BCELoss().cuda() # BCE only support the input in a range -1 < x < 1
        except :
            self.reconst_error = nn.BCELoss()

    def __call__(self, output, target, mu, sigma):
        mrg_likelihood = torch.mean(torch.sum(target * torch.log(output) + (1 - target) * torch.log(1 - output), dim=1))
        # mrg_likelihood = self.reconst_error(output, target) # (output, target)
        kl_div = self.KLD(mu, sigma)
        
        ELBO = mrg_likelihood - kl_div
        loss = -ELBO

        return loss, -mrg_likelihood, kl_div

    @staticmethod
    def KLD(mu, sigma):
        res = 0.5 * torch.sum(torch.pow(mu, 2) + torch.pow(sigma, 2) - torch.log(1e-8 + torch.pow(sigma,2)) - torch.ones_like(mu), dim=1)
        res = torch.mean(res)
        # res = (mu ** 2) + (sigma ** 2 ) - torch.log(1e-8 + (sigma ** 2)) - 1   
        # res = torch.mean(0.5 * torch.sum(res, dim=1))

        return res