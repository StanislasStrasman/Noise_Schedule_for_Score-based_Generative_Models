import numpy as np
import torch
from scipy.stats import multivariate_normal
from scipy.special import logsumexp

class MG25:
    def __init__(self, d):
        self.d = d
        self.cov_matrix = self.make_cov_matrix()
        self.means = self.make_means()

    def make_cov_matrix(self):
        diag_elements = [0.01, 0.01] + [0.1] * (self.d - 2)
        return np.diag(diag_elements)

    def make_means(self):
        means = []
        for i in range(-2, 3):
            for j in range(-2, 3):
                mean = [i, j] + [0] * (self.d - 2)
                means.append(mean)
        return np.array(means)

    def sample(self, n_samples):
        samples = np.empty((n_samples, self.d))  
        mean_indices = np.random.choice(len(self.means), size=n_samples)  
        for index, mean_index in enumerate(mean_indices):
            mu = self.means[mean_index]
            sample = np.random.multivariate_normal(mean=mu, cov=self.cov_matrix)
            samples[index] = sample  
        return torch.tensor(samples, dtype=torch.float32)       

    def get_log_pdf(self, x):
        x = np.asarray(x)
        log_pdf_values = []
        for mean in self.means:
            log_pdf_values.append(multivariate_normal.logpdf(x, mean=mean, cov=self.cov_matrix))
        normalized_log_pdf = logsumexp(log_pdf_values) - np.log(25)
        return normalized_log_pdf

    def compute_NLL(self, samples):
        log_pdf_values = np.array([self.get_log_pdf(sample) for sample in samples])
        nll = -np.mean(log_pdf_values)
        return nll


class Funnel:
    def __init__(self, d, a=1, b=0.5):
        self.d = d
        self.a = a
        self.b = b

    def sample(self, n_samples):
        x1 = np.random.normal(0, self.a, size=n_samples)
        samples = np.zeros((n_samples, self.d))
        samples[:, 0] = x1
        variances = np.exp(2 * self.b * x1)
        samples[:, 1:] = np.random.normal(0, np.sqrt(variances)[:, np.newaxis], size=(n_samples, self.d - 1))
        return torch.tensor(samples, dtype=torch.float32)

    def get_log_pdf(self, x):
        x = np.asarray(x)
        cov_x1 = self.a**2
        cov_rest = np.exp(2 * self.b * x[0])
        log_pdf_x1 = multivariate_normal.logpdf(x[0], mean=0, cov=cov_x1)
        log_pdf_rest = sum(multivariate_normal.logpdf(x[i], mean=0, cov=cov_rest) for i in range(1, len(x)))
        return log_pdf_x1 + log_pdf_rest
 
    def compute_NLL(self, samples):
        log_pdf_values = np.array([self.get_log_pdf(sample) for sample in samples])
        nll = -np.mean(log_pdf_values)
        return nll











