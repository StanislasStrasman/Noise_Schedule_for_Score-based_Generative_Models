import torch
import math
import tqdm
import numpy as np
import functions as func
from torch.optim import Adam
from decoder import Decoder
from upperbounds import gaussian
from tqdm import tqdm, trange

#######################
### Diffusion processes
#######################
'''
Classes associated with the implementation of forward stochastic differential equations. See documentation for more information. 
'''
class forward_sde: 
    def __init__(self, dimension, final_time, sigma_infty, device=torch.device('cpu')):

        self.d = dimension
        self.final_time = final_time
        self.sigma_infty = sigma_infty
        self.device = device
        self.final = gaussian(dimension, 
                              torch.zeros(dimension, device = device), 
                              self.sigma_infty**2 * torch.eye(dimension, device = device))
    def to(self, device):
        self.device = device
        self.final = self.final.to(device)

class forward_VPSDE(forward_sde):
    def __init__(self, dimension, beta, sigma_infty, final_time, device=torch.device('cpu')):
        super().__init__(dimension, final_time, sigma_infty, device)  
        self.beta = beta
        self.sigma_infty = sigma_infty
    def alpha(self, time_t):
        return self.beta(time_t) / (2 * self.sigma_infty**2) 
    def eta(self, time_t):
        return torch.sqrt(self.beta(time_t))
    def alpha_integrate(self, time_t): 
        return self.beta.integrate(time_t) / (2 * self.sigma_infty**2) 
    def sigma(self, time_t):
        return self.sigma_infty * torch.sqrt(1. - torch.exp(- 2*self.alpha_integrate(time_t)))
    def mu(self, time_t):
        return torch.exp(-self.alpha_integrate(time_t))
    def I1(self, time_t):
        return 2*self.sigma_infty**2 * (torch.exp(self.alpha_integrate(time_t)) - 1.)
    def I2(self, time_t):
        return self.sigma_infty**2 * (torch.exp(2*self.alpha_integrate(time_t)) - 1.)

#####################
### Noising functions
#####################
'''
Classes associated with the family of parametrics noise schedule. See documentation for more information. 
'''
class beta_parametric:
    def __init__(self, a, final_time, beta_min, beta_max):
        self.a = a
        self.final_time = final_time
        self.beta_min = beta_min
        self.beta_max = beta_max
        if a == 0:
            self.delta = (beta_max - beta_min) / final_time
        else: 
            self.delta = (beta_max - beta_min) / (math.exp(self.a * final_time) - 1.)
    def __call__(self, t):
        if np.abs(self.a) < 1e-3: # à changer
            return self.beta_min + self.delta * t
        else:
            return self.beta_min + self.delta * (torch.exp(self.a*t) - 1.)
    def integrate(self, t): 
        if np.abs(self.a) < 1e-3:
            return self.beta_min * t + 0.5 * self.delta * t**2
        else:
            return self.beta_min * t + self.delta * ((torch.exp(self.a*t)-1)/self.a - t)
    def square_integrate(self,t):
        if np.abs(self.a) < 1e-3:
            return self.beta_min**2 * t +  self.beta_min * self.delta * t**2 + (1./3) * self.delta**2 * t**3  #modified
        else:
            res = self.beta_min**2 * t + 2*self.beta_min*self.delta*(torch.exp(self.a*t) / self.a - t) 
            res += (self.delta)**2 * ( (torch.exp(2*self.a*t))/(2* self.a) - 2* (torch.exp(self.a*t))/(self.a) + t)
            res -= (2* self.beta_min * self.delta /self.a - self.delta**2 *(3/2)*(1/self.a))
            return res  
    def change_a(self, a): 
        self.a = a 
        if np.abs(self.a) < 1e-3: 
            self.delta = (self.beta_max - self.beta_min) / self.final_time 
        else:
            self.delta = (self.beta_max - self.beta_min) / (math.exp(self.a * self.final_time) - 1.)

class beta_cosine:
    def __init__(self, final_time, beta_min, clip_max = None):
        self.final_time = final_time
        self.beta_min = beta_min
        self.clip_max = clip_max


    def __call__(self, t):
        sched_val = np.pi * np.tan(np.pi * (self.beta_min + t / self.final_time) / (2 * (self.beta_min + 1))) / (self.final_time * (self.beta_min + 1))
        if self.clip_max is not None:
            sched_val = min(sched_val, self.clip_max)
        return sched_val

     
    def integrate(self, t):
        h_t = np.cos(np.pi * (self.beta_min + t / self.final_time) / (2 * (self.beta_min + 1)))**2
        h_0 = np.cos(np.pi * self.beta_min / (2 * (self.beta_min + 1)))**2
        integral_beta = -np.log(h_t / h_0) 
        return integral_beta

############
### Training
############

def generate_forward(sde, x0, time_tau):
    mean = sde.mu(time_tau) * x0
    noise = sde.sigma(time_tau) * torch.randn_like(x0, device = sde.device)
    return mean + noise, noise
    
class loss_explicit:
    def __init__(self, score_theta, sde, score_explicit, eps=1e-5):
        self.score_theta = score_theta
        self.score_explicit = score_explicit
        self.sde = sde
        self.eps = eps
    def __call__(self, x0):
        time_tau = torch.rand((x0.shape[0], 1), device=x0.device) * (self.sde.final_time - self.eps) + self.eps
        x_tau, noise = generate_forward(self.sde, x0, time_tau)
        score = self.score_theta(x_tau, time_tau)
        target = self.score_explicit(x_tau, time_tau)
        loss = torch.mean(torch.sum((score - target)**2, axis=1))
        return loss

## compute the explicit score in the Gaussian case
class explicit_score:
    def __init__(self, sde, dataset):
        self.mu_0, self.sigma_0 = dataset.mean_covar()
        self.sde = sde
        self.id_d = torch.eye(self.sde.d, device = self.sde.device)  

    def __call__(self, x, t): 
        if len(t) == 1:
            t = torch.full((x.shape[0], 1), t.item(), device=self.sde.device)
        mu_t, sigma_t = self.sde.mu(t), self.sde.sigma(t)
        mat = torch.inverse((mu_t**2).unsqueeze(-1) * self.sigma_0 
                            + (sigma_t**2).unsqueeze(-1) * self.id_d)
        score = -torch.bmm(mat, (x - mu_t * self.mu_0).unsqueeze(-1))
        return score.squeeze(-1)

class loss_conditional:
    def __init__(self, score_theta, sde, eps=1e-5):
        self.score_theta = score_theta
        self.sde = sde
        self.eps = eps
    def __call__(self, x0): 
        time_tau = torch.rand((x0.shape[0], 1), device=x0.device) * (self.sde.final_time - self.eps) + self.eps
        x_tau, noise = generate_forward(self.sde, x0, time_tau)     
        score = self.sde.sigma(time_tau)**2 * self.score_theta(x_tau, time_tau)
        target = - noise 
        loss = torch.mean(torch.sum((score - target)**2, axis=1))
        return loss        

def train(loss_fn, dataloader, n_epochs, optimizer):
    tqdm_epoch = trange(n_epochs)
    for epoch in tqdm_epoch:
        avg_loss = 0.
        num_items = 0
        for x0 in dataloader:
            
            loss = loss_fn(x0) 
            optimizer.zero_grad()
            loss.backward()    
            optimizer.step()
            avg_loss += loss.item() * x0.shape[0]
            num_items += x0.shape[0]
        tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))


