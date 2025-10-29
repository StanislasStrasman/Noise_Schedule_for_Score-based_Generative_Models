import numpy as np
import torch
import math
import scipy
from scipy.spatial import KDTree

from . import functions as func


#########################
### GAUSSIAN DISTRIBUTION 
#########################
'''
Class associated with Gaussian distribution and its relevant constants (Lipschitz, log-concavity).
'''
class gaussian:
    def __init__(self, dimension, mu, sigma):
        self.d = dimension
        self.device = mu.device
        self._mu = mu
        self._sigma = sigma
        self._sq_sigma = torch.linalg.cholesky(self._sigma)
    def mean_covar(self,):
        return self._mu, self._sigma
    def generate_sample(self, size): # Returns: torch.Tensor  [size , dimension]
        self.sample = self._mu + torch.randn(size, self.d, device=self.device) @ self._sq_sigma.T
        return self.sample
    def to(self, device):
        self.device = device
        self._mu = self._mu.to(device)
        self._sigma = self._sigma.to(device)
        self._sq_sigma = self._sq_sigma.to(device)
        if "sample" in self.__dict__:
            self.sample = self.sample.to(device)
    def compute_C0(self):
        # Compute the smallest eigenvalue of the Hessian of the Gaussian density (i.e. the inverse of largest eigenvalue of covariance matrix)
        eigenvalues = torch.linalg.eigvals(self._sigma)
        largest_eigenvalue = torch.max(torch.abs(eigenvalues))
        C0 = 1.0 / largest_eigenvalue
        return C0
    def compute_L0(self): 
        # Compute the largest eigenvalue of the Hessian of the Gaussian density (i.e. the inverse of smallest eigenvalue of covariance matrix)
        eigenvalues = torch.linalg.eigvals(self._sigma)
        smallest_eigenvalue = torch.min(torch.abs(eigenvalues))
        L0 = 1.0 / smallest_eigenvalue
        return L0
'''
Function associated with Proposition D.1.
'''
def compute_Ct(dataset, sde, t, gaussian = True):
    if (gaussian == True):
        cov = dataset._sigma
    else:
        cov = compute_cov_matrix(dataset)
    eigenvalues = torch.linalg.eigvals(cov)
    lambda_max = torch.max(torch.abs(eigenvalues))
    over = sde.mu(t)**2 * (sde.sigma_infty**2 - lambda_max)
    under = sde.mu(t)**2 * lambda_max + sde.sigma_infty**2 * (1 - sde.mu(t)**2)
    return over / under

'''
Function associated with Proposition D.2.
'''
def compute_Lt(dataset, sde, t, gaussian=True):
    if gaussian:
        L_0 = dataset.compute_L0()
    else:
        cov = empirical_mean_covar(dataset)[1]
        smallest_eigenvalue = torch.min(torch.abs(torch.linalg.eigvals(cov)))
        L_0 = 1.0 / smallest_eigenvalue

    v1 = 1.0 / (sde.sigma_infty**2 * (1 - sde.mu(t)**2))
    v2 = L_0 / (sde.mu(t)**2)
    return torch.minimum(v1, v2) - (1.0 / sde.sigma_infty**2)

#################################
### COMPUTE THE MIXING TIME ERROR
#################################

def compute_C1(dataset, sde):
    return torch.tensor([ func.kl(dataset, sde.final), func.w2(dataset, sde.final) ])

#KL UPPER BOUND MIXING
def compute_E1(dataset,sde):
    C1, _ = compute_C1(dataset, sde)
    return C1 * torch.exp(-2*sde.alpha_integrate(torch.tensor(sde.final_time)))   

def compute_E1_no_contraction(dataset,sde):
    C1, _ = compute_C1(dataset, sde)
    return C1 * torch.exp(-sde.alpha_integrate(torch.tensor(sde.final_time)))

#W2 UPPER BOUND MIXING
def compute_mixing_w2(dataset,sde):
    _, w2_data_invariant = compute_C1(dataset, sde)
    mixing = w2_data_invariant * torch.exp(- (1/sde.sigma_infty**2) * sde.beta.integrate(torch.tensor(sde.final_time)))
    return mixing

##########################################################
### COMPUTE THE APPROXIMATION ERROR & DISCRETIZATION ERROR
##########################################################

# APPROXIMATION ERROR
def compute_C2(dataset, sde, score_theta, true_score, num_steps, num_mc):
    times = torch.linspace(0, sde.final_time, num_steps+1, device=sde.device) 
    result = 0.
    result_sup_L2 = 0.
    for i in range(len(times) - 1):
        rev_tk =  sde.final_time - times[i]
        rev_tkp1 = sde.final_time - times[i+1]
        x0 = dataset.generate_sample(num_mc)
        x_revtk = sde.mu(rev_tk) * x0 + sde.sigma(rev_tk) * torch.randn_like(x0, device = sde.device)

        diff = score_theta(x_revtk, rev_tk) - true_score(x_revtk, rev_tk.unsqueeze(-1)) 
        M = torch.mean(torch.sum(diff**2, axis=1)).item() 
        result += M * (sde.beta.integrate(rev_tk) - sde.beta.integrate(rev_tkp1 ))
        if (result_sup_L2 < M):
            result_sup_L2 = np.sqrt(M)
    return result, result_sup_L2

def compute_E2(dataset, sde, score_theta, true_score, num_steps, num_mc):
    C2 = compute_C2(dataset, sde, score_theta , true_score, num_steps, num_mc)
    return C2

#KL DISCRETIZATION ERROR
def compute_E3(dataset, sde, num_steps):
    sigma_infty = sde.sigma_infty
    step_size = sde.final_time / num_steps
    fisher_info = func.relative_fisher_information_Gaussian(dataset, sde)
    beta_T = sde.beta(torch.tensor(sde.final_time))
    E3 = 2*( step_size * beta_T ) * torch.max(step_size * beta_T / (4 * sigma_infty**2), torch.tensor(1.0)) * fisher_info
    return E3


def compute_ellbar(dataset, training_sample, sde, num_steps, gauss = True):
    times = torch.linspace(0, sde.final_time, num_steps+1, device = sde.device) 
    hist= []
    if gauss == True:
        eigen = torch.abs(torch.linalg.eigvals(dataset._sigma))
    else:
        sigma = func.empirical_mean_covar(training_sample)[1] 
        eigen = torch.abs(torch.linalg.eigvals(sigma))
        
    lambda_min = torch.min(eigen)
    lambda_max = torch.max(eigen)
    for i in range(len(times) - 1):
        tk = times[i]
        tkp1 = times[i+1]
        k_1_over =  sde.beta(tkp1)/(sde.sigma_infty**2)*sde.mu(tk)**2  * torch.abs(lambda_min - sde.sigma_infty**2)
        k_1_under = torch.abs( (sde.sigma_infty**2 + sde.mu(tk)**2 * (lambda_min - sde.sigma_infty**2)) \
                        * (sde.sigma_infty**2 + sde.mu(tkp1)**2 * (lambda_min - sde.sigma_infty**2)))
        kappa_1 = k_1_over/k_1_under
        norm_mu = torch.norm(torch.mean(training_sample, axis = 0),p=2)
        M = (sde.beta(tkp1)/(2*sde.sigma_infty**2))*sde.mu(tk)
        kappa_2_over = norm_mu  * M*torch.abs(sde.mu(tk)*sde.mu(tkp1)*(lambda_min -sde.sigma_infty**2 ) - sde.sigma_infty**2)
        kappa_2 = kappa_2_over / k_1_under
        hist.append( np.max([kappa_1,kappa_2]))                                                               
    return np.max(hist)
    
#COMPLETE W2 BOUND 
def compute_w2_bound(dataset, training_sample, sde, num_steps, epsilon, gauss = True):
    #constants computation
    h = 1/num_steps
    T = sde.final_time
    B = torch.sqrt(func.L2_norm_estimator(training_sample)**2 + sde.sigma_infty**2 * sde.d)
    beta_final = sde.beta(torch.tensor(sde.final_time))
    M_2 = torch.sqrt(2*h* beta_final)/sde.sigma_infty + h*beta_final/(2* sde.sigma_infty**2)

    if gauss == True: #to evaluate the upper bound when the target distribution is Gaussian
        ellbar = compute_ellbar(dataset, training_sample, sde, num_steps, gauss)
        #mixing
        mixing =  compute_mixing_w2(dataset,sde)
        t_points = torch.linspace(0, sde.final_time, steps=100, device = sde.device)
        ct_values = torch.tensor([compute_Ct(dataset, sde, t) * sde.beta(t) for t in t_points])
        integral_approximation_Ct = torch.trapezoid(ct_values, t_points)
        mixing *= torch.exp(- integral_approximation_Ct)

        #appprox+discr
        aprox_discr = 0
        times = torch.linspace(0, sde.final_time, num_steps+1, device = sde.device) 

        for i in range(len(times) - 1):
            rev_tk = sde.final_time - times[i]  #T-tk
            rev_tkp1 = sde.final_time - times[i+1]  #T-tkp1
            t_points = torch.linspace(rev_tkp1, rev_tk, steps=100, device = sde.device)
            Lt_beta_values = torch.tensor([compute_Lt(dataset, sde, t) * sde.beta(t) for t in t_points])
            integral_approximation_Lt_beta = torch.trapezoid(Lt_beta_values, t_points)
            aprox_discr += integral_approximation_Lt_beta * (M_2 + 2*integral_approximation_Lt_beta)*B

        const_2 = epsilon* T * beta_final
        const_3 = ellbar*h* T * beta_final* (1 + 2 * B ) 

    else: # to approximate the upper bound in the general case
        empirical_covariance = func.empirical_mean_covar(training_sample)[1] #compute_cov_matrix(training_sample)
        empirical_mean = torch.mean(training_sample, dim=0)
        dataset = gaussian(sde.d,empirical_mean, empirical_covariance)
        ellbar = compute_ellbar(dataset, training_sample, sde, num_steps, True)

        #mixing
        mixing =  compute_mixing_w2(dataset,sde)
        t_points = torch.linspace(0, sde.final_time, steps=100, device = sde.device)
        ct_values = torch.tensor([compute_Ct(dataset, sde, t) * sde.beta(t) for t in t_points])
        integral_approximation_Ct = torch.trapezoid(ct_values, t_points)
        mixing *= torch.exp(- integral_approximation_Ct)

        #appprox+discr
        aprox_discr = 0
        times = torch.linspace(0, sde.final_time, num_steps+1, device = sde.device) 
        for i in range(len(times) - 1):
            rev_tk = sde.final_time - times[i] #T-tk
            rev_tkp1 = sde.final_time - times[i+1] #T-tkp1
            t_points = torch.linspace(rev_tkp1, rev_tk, steps=100, device = sde.device)
            Lt_beta_values = torch.tensor([compute_Lt(dataset, sde, t) * sde.beta(t) for t in t_points])
            integral_approximation_Lt_beta = torch.trapezoid(Lt_beta_values, t_points)
            aprox_discr += integral_approximation_Lt_beta * (M_2 + 2*integral_approximation_Lt_beta)*B

        const_2 = 0
        const_3 = ellbar*h* T * beta_final* (1 + 2 * B ) 
    
    return mixing + aprox_discr + const_2 + const_3