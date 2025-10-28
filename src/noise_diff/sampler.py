import torch
import math
from . import diffusion as diff

#######################################
### DISCRETIZATION OF THE BACKWARD SDE 
######################################

def Euler_Maruyama_discr_sampler(init, sde, score_theta, num_steps):
    x_bar = init.clone()
    step_size = sde.final_time / num_steps
    with torch.no_grad():
        time_T = torch.tensor([sde.final_time], dtype=init.dtype, device=init.device)
        for n in range(1, num_steps+1):
            rev_tn = time_T - n * step_size
            drift = sde.alpha(rev_tn) * x_bar + sde.eta(rev_tn)**2 * score_theta(x_bar, rev_tn)
            noise = sde.eta(rev_tn) * math.sqrt(step_size) * torch.randn_like(init)
            x_bar += drift * step_size + noise
    return x_bar

def EI_discr_sampler(init, sde, score_theta, num_steps):
    x_bar = init.clone()
    step_size = sde.final_time / num_steps
    with torch.no_grad():
        time_T = torch.tensor([sde.final_time], dtype=init.dtype, device=init.device)
        for n in range(0, num_steps):
            rev_tn = time_T - n * step_size
            rev_tnp1 = rev_tn - step_size
            delta = sde.alpha_integrate(rev_tn) - sde.alpha_integrate(rev_tnp1)
            J1 = torch.exp(-sde.alpha_integrate(rev_tnp1)) * (sde.I1(rev_tn) - sde.I1(rev_tnp1))
            J2 = torch.exp(-2*sde.alpha_integrate(rev_tnp1)) * (sde.I2(rev_tn) - sde.I2(rev_tnp1))
            noise = torch.sqrt(J2) * torch.randn_like(init)
            x_bar = torch.exp(delta) * x_bar + score_theta(x_bar, rev_tn) * J1 + noise 
    return x_bar


