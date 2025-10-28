# Short Documentation of the Main Objects in This Repository

# `diffusion.py` : forward SDE & noise schedules.

## `forward_SDE` Class Documentation

```python
import torch
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
```

The `forward_sde` class provides a base implementation for stochastic differential equations (SDEs). It serves as a foundation for specific forward SDEs such as the Ornstein–Uhlenbeck process, also known as the **Variance-Preserving SDE (VPSDE)** .
### **Attributes**

| Attribute       | Type           | Description                                                                                         |
|-----------------|----------------|-----------------------------------------------------------------------------------------------------|
| `d`            | `int`          | Dimension of the state space.                                        |
| `final_time`   | `float`        | Diffusion time $T$ of the SDE.                                       |
| `sigma_infty`  | `float`        | Asymptotic standard deviation. |
| `device`       | `torch.device`       | Computational device (e.g., `'cpu'` or `'cuda'`).                           |


### **Methods**

#### `__init__(self, dimension, final_time, sigma_infty, device=torch.device('cpu'))`

Initializes the `forward_sde` class.

#### `to(self, device)`

Transfers the object to the specified computational device.

---

## `forward_VPSDE` Class Documentation

```python
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
```

The `forward_VPSDE` class implements a **Variance-Preserving Stochastic Differential Equation (VPSDE)**. This process is commonly used in Score-based Generative Models, where the variance of the system evolves in a controlled manner throughout diffusion as opposed to **VESDE**.
The class inherits from the `forward_sde` base class and introduces a scalar time-varying noise schedule $\beta(t)$.

### **Key Mathematical Characteristics**

The **Variance-Preserving SDE** is defined as:
$$
\mathrm{d} \overrightarrow{X}_t = -\frac{1}{2 \sigma^2} \beta(t) \overrightarrow{X}_t \, \mathrm{d} t + \sqrt{\beta(t)} \mathrm{d} B_t
$$
where:
- $\beta(t):[0,T] \to \mathbb{R}_+ $: time-dependent noise schedule.
- $(B_t)_{t \in [0,T]}$: standard Brownian motion in $\mathbb{R}^d$.
- $\sigma^2$: variance of the the stationary distribution of the process.


### **Attributes**

| Attribute       | Type             | Description                                                                                       |
|-----------------|------------------|---------------------------------------------------------------------------------------------------|
| `beta`         | `callable`       | Time-dependent noise coefficient  $\beta(t)$.                                                  |
| Inherited      | From `forward_sde` | Attributes like `d`, `final_time`, `sigma_infty`, and `device`.                                  |

### **Methods**

| Method                       | Description                                                                                     |
|------------------------------|-------------------------------------------------------------------------------------------------|
| `alpha(self, time_t)`        | Returns the drift coefficient at time $t \in [0,T]$.           |
| `eta(self, time_t)`          | Returns the diffusion coefficient value at time $t \in [0,T]$.                                   |
| `alpha_integrate(self, time_t)` | Computes the integrated drift coefficient between time 0 and $t$.              |
| `sigma(self, time_t)`        | Computes the standard deviation of the VPSDE process at time $t  \in [0,T]$. |
| `mu(self, time_t)`           | Computes the mean decay factor of the VPSDE process at time $t \in [0,T]$.            |
| `I1(self, time_t)`           | Computes integrated quantities for EI integrator scheme (see below for details). |
| `I2(self, time_t)`           | Computes integrated quantities for EI integrator scheme (see below for details). |



### **Detailed Method Descriptions**

The class inherits from the `forward_sde` base class and introduces a time-varying noise schedule $\beta(t) $. Here, $\sigma_{\infty}$ corresponds to the attribute `self.sigma_infty`, which represents the asymptotic standard deviation of the Gaussian stationary distribution of the OU process.

#### `alpha(self, time_t)`

**Description:**  
Returns the drift coefficient value at time $t \in [0,T]$:
$$ \alpha(t) = \frac{\beta(t)}{2\sigma_{\infty}^2}.
$$
**Parameters:**
- `time_t` *(float or tensor)*: time $t$.

**Returns:**
- $\alpha(t)$ *(float or tensor)*.

---

#### `eta(self, time_t)`

**Description:**  
Returns the diffusion coefficient value at time $t \in [0,T]$:
$$
\eta(t) = \sqrt{\beta(t)}.
$$

**Parameters:**
- `time_t` *(float or tensor)*: time $t$.

**Returns:**
- $\eta(t)$ *(float or tensor)*.

---

#### `alpha_integrate(self, time_t)`

**Description:**  
Computes the integrated drift coefficient between time 0 and $t$:  
$$
\int_0^t \alpha(t) \, \mathrm{d}s = \frac{1}{2\sigma_{\infty}^2} \int_0^t \beta(s) \, \mathrm{d}s.
$$

**Parameters:**
- `time_t` *(float or tensor)*: time $t$.

**Returns:**
- $\int_0^t \alpha(t) \, \mathrm{d}s)$ *(float or tensor)*.

---

#### `sigma(self, time_t)`

**Description:**  
Computes the standard deviation $\sigma (t)$ of the VPSDE process at time $t  \in [0,T]$:
$$
\sigma(t) = \sigma_{\infty} \sqrt{1 - \exp\left(-2 \int_0^t \alpha(s) \, \mathrm{d}s \right)}
$$

**Parameters:**
- `time_t` *(float or tensor)*: Time $t$.

**Returns:**
- $\sigma(t)$ *(float or tensor)*: Standard deviation at time $t$.

---

#### `mu(self, time_t)`

**Description:**  
Computes the mean decay factor $\mu(t)$ of the VPSDE process at time $t \in [0,T]$. 
$$
\mu(t) = \exp\left(-\int_0^t \alpha(s) \, \mathrm{d}s \right)
$$

**Parameters:**
- `time_t` *(float or tensor)*: Time $t $.

**Returns:**
- $\mu(t)$ *(float or tensor)*: Mean decay factor at time $t$.

---

#### `I1(self, time_t)`

**Description:**  
Computes the integral $I_1(t)$ (needed in the Exponential Integrator scheme):
$$
I_1(t) = 2\sigma_{\infty}^2 \left(e^{\int_0^t \alpha(s) \mathrm{d}s} - 1\right).
$$

**Parameters:**
- `time_t` *(float or tensor)*: time $t$.

**Returns:**
- $I_1(t)$ *(float or tensor)*.

---

#### `I2(self, time_t)`

**Description:**  
Computes the integral $I_2(t)$ (needed in the Exponential Integrator scheme):
$$
I_2(t) = \sigma_{\infty}^2 \left(e^{2\int_0^t \alpha(s) \mathrm{d}s} - 1\right).
$$

**Parameters:**
- `time_t` *(float or tensor)*: time $t$.

**Returns:**
- $I_2(t)$ *(float or tensor)*.

---

## `generate_forward` Function Documentation

```python
def generate_forward(sde, x0, time_tau):
    mean = sde.mu(time_tau) * x0
    noise = sde.sigma(time_tau) * torch.randn_like(x0, device = sde.device)
    return mean + noise, noise
```


The `generate_forward` function generates a sample from the **forward stochastic process**. Given an initial state $X_0$, it computes the forward step at a specified time $\tau \in [0,T]$. The function uses the **mean** and **noise** terms of the SDE to simulate the forward state.



### **Mathematical Definition**

Given an initial state $X_0 \in \mathbb{R}^d$, $Z \sim \mathcal{N}(0, I_d)$, $Z \perp X_0$, the forward step at time $\tau$ is given by:
$$
X_\tau = \mu(\tau) x_0 + \sigma(\tau) Z.
$$

---


## `explicit_score` Class Documentation

```python
class explicit_score:
    def __init__(self, sde, dataset):
        self.mu_0, self.sigma_0 = dataset.mean_covar()
        self.sde = sde
        self.id_d = torch.eye(self.sde.d, device=self.sde.device)

    def __call__(self, x, t):
        if len(t) == 1:
            t = torch.full((x.shape[0], 1), t.item(), device=self.sde.device)
        mu_t, sigma_t = self.sde.mu(t), self.sde.sigma(t)
        mat = torch.inverse(
            (mu_t**2).unsqueeze(-1) * self.sigma_0
            + (sigma_t**2).unsqueeze(-1) * self.id_d
        )
        score = -torch.bmm(mat, (x - mu_t * self.mu_0).unsqueeze(-1))
        return score.squeeze(-1)
```
The `explicit_score` class provides a **closed-form (orcale) score function** for the forward process marginal $p_t(x)$ when the initial data distribution is given by $X_0 \sim \mathcal{N}(\mu_0,\Sigma_0)$ (see Lemma E.1 in the paper).

### **Attributes**

| Parameter | Type | Description |
|------------|------|-------------|
| `sde` | `forward_sde` or subclass | Provides the functions $\mu(t)$, $\sigma(t)$, the dimension `d`, and the computational `device`. |
| `dataset` | object | Must implement `mean_covar()` returning `(mu_0, sigma_0)` with a mean vector `mu_0` of shape `(d,)` and a covariance matrix `sigma_0` of shape `(d, d)`. |

---

## `loss_conditional` Class Documentation

```python
class loss_conditional:
    def __init__(self, score_theta, sde, eps=1e-5):
        self.score_theta = score_theta
        self.sde = sde
        self.eps = eps

    def __call__(self, x0):
        time_tau = torch.rand((x0.shape[0], 1), device=x0.device) * (self.sde.final_time - self.eps) + self.eps
        x_tau, noise = generate_forward(self.sde, x0, time_tau)
        score = self.sde.sigma(time_tau)**2 * self.score_theta(x_tau, time_tau)
        target = -noise
        loss = torch.mean(torch.sum((score - target)**2, axis=1))
        return loss
```


The `loss_conditional` class defines the **conditional denoising score-matching loss** used to train the score network $s_\theta : [0,T] \times \mathbb{R}^d \to \mathbb{R}^d$ in **Score-based Generative Models (SGMs)**.  



### **Attributes**

| Attribute | Type | Description |
|------------|------|-------------|
| `score_theta` | `callable` | Neural network approximating the score function $\nabla_x \log p_t(x) $. |
| `sde` | `forward_sde` or subclass | Defines the forward diffusion process used to generate perturbed samples $X_\tau$. |
| `eps` | `float` | Small positive constant  for numerical stability to avoid sampling times too close to 0. |



### **Methods**

| Method | Description |
|---------|-------------|
| `__init__(self, score_theta, sde, eps=1e-5)` | Initializes the loss function with the given score model, SDE, and numerical cutoff. |
| `__call__(self, x0)` | Computes the denoising score-matching loss for a batch of clean samples $ x_0 \sim X_0$. |


### **Mathematical Definition**

At each call, a random time $\tau \sim \mathcal{U}(\varepsilon, T)$ independent of $X_0$ is sampled, and the noisy sample

$$
X_\tau = \mu(\tau) X_0 + \sigma(\tau) Z, \quad Z \sim \mathcal{N}(0, I_d),
$$
is generated. The objective minimized by `loss_conditional`, using a positive weighting function $\lambda(\tau) = \sigma^2(\tau)$ that controls the contribution of each diffusion time to the overall objective, yielding  
$$
\mathcal{L}(\theta)
= \mathbb{E}_{X_0, Z, \tau} \bigg[ \big\| \sigma(\tau)^2 s_\theta(X_\tau, \tau) + Z \big\|^2 \bigg].
$$



### **Returns**

| Type | Description |
|------|-------------|
| `torch.Tensor` | Scalar loss value corresponding to the batch-averaged denoising score-matching objective. |

---


## Noise schedule choices

## `beta_parametric` Class Documentation

```python
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
        if np.abs(self.a) < 1e-3: #for numerical stability
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
```

The `beta_parametric` class implements a parametric noise schedule $\beta_a(t)\in \mathbb{R}_+$ for $a \in \mathbb{R}$,
$$ \beta_a(t) \propto (\mathrm{e}^{at} -1)/(\mathrm{e}^{aT} -1).$$
The parameter $a$, determines determines the convexity of the curve between its initial value $\beta_{\min}$ and its terminal value $\beta_{\max}$ over the interval $[0,T]$.Note in particular that it provides convex function when $a>0$, concave function when $a<0$ and linear function when $a=0$. In addition, it provides methods to compute integrals of $\beta(t)$ or $\beta^2(t)$. 


 ## **Attributes**

| Attribute       | Type    | Description                                                                                           |
|-----------------|---------|-------------------------------------------------------------------------------------------------------|
| `a`             | `float` | Determines whether the convexity of $\beta_a(t)$.  |
| `final_time`    | `float` | Diffusion time $T$.                                                                               |
| `beta_min`      | `float` | Initial value of $\beta_a$ (i.e. $\beta(0)$ is diffusion starts at time $0$).                                                          |
| `beta_max`      | `float` | Final value of $\beta_a$ (i.e. $\beta_a(T)$).                                                            |


## **Methods**

 | Method                                | Description                                                                                                     |
|---------------------------------------|-----------------------------------------------------------------------------------------------------------------|
| `__init__(self, a, final_time, beta_min, beta_max)` | Initializes the schedule.                 |
| `__call__(self, t)`                  | Returns  $\beta_a(t)$ at a given time $t$.                                               |
| `integrate(self, t)`                 | Computes $\int_0^t \beta_a(s) \, \mathrm{d} s$.                                                                           |
| `square_integrate(self, t)`          | Computes $\int_0^t \beta_a^2(s) \,   \mathrm{d} s$.                                                                         |
| `change_a(self, a)`                  | Updates the parameter `a`.                                                |

---

## `beta_cosine` Class Documentation

```python
class beta_cosine:
    def __init__(self, final_time, beta_min, sigma2 = 1, clip_max = None):
        self.final_time = final_time
        self.beta_min = beta_min
        self.clip_max = clip_max 
        self.sigma2 = sigma2



    def __call__(self, t):
        sched_val = self.sigma2 * np.pi * np.tan(np.pi * (self.beta_min + t / self.final_time) / (2. * (self.beta_min + 1.))) / (self.final_time * (self.beta_min + 1.))
        if self.clip_max is not None:
            sched_val = min(sched_val, self.clip_max)
        return sched_val

     
    def integrate(self, t):
        h_t = np.cos(np.pi * (self.beta_min + t / self.final_time) / (2. * (self.beta_min + 1.)))**2.
        h_0 = np.cos(np.pi * self.beta_min / (2 * (self.beta_min + 1)))**2
        integral_beta = -2* self.sigma2 * np.log(h_t / h_0)
        return integral_beta
```

The `beta_cosine` class implements a noise schedule $\beta_{\text{cos}}(t)$ based on a **cosine/tangent** relationship adapted from Improved Denoising Diffusion Probabilistic Models (Nichol & Dhariwal, 2021) to the VPSDE framework.
     $$
       \beta_{\text{cos}}(t) = \sigma^2 \frac{\pi}{T \times (\beta_{\min} + 1)} \,\tan\!\Bigl(\frac{\pi}{2}\,\frac{\beta_{\min} + \tfrac{t}{T}}{\beta_{\min} + 1}\Bigr).
     $$


## **Attributes**

| Attribute     | Type      | Description                                                                               |
|---------------|-----------|-------------------------------------------------------------------------------------------|
| `final_time`  | `float`   | Diffusion time $T$.                                                                   |
| `beta_min`    | `float`   | A parameter that shifts the cosine schedule.                                      |
| `clip_max`    | `float` or `None` | Maximum allowed value for $\beta(t)$. If `None`, no clipping is applied.                   |


## **Methods**

| Method                                    | Description                                                                         |
|-------------------------------------------|-------------------------------------------------------------------------------------|
| `__init__(self, final_time, beta_min, clip_max=None)` | Initializes the cosine-based noise schedule.                     |
| `__call__(self, t)`                      | Returns the noise coefficient $\beta_{\text{cos}}(t)$ at time $t$.                           |
| `integrate(self, t)`                     | Computes $\int_0^t \beta(s)\, \mathrm{d}s$.                                               |

---


# `sampler.py` : discretization schemes for the backward process.

## `Euler_Maruyama_discr_sampler` Function Documentation


```python
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
```

The `Euler_Maruyama_discr_sampler` function implements the **discrete-time Euler-Maruyama method** to approximate samples from the **reverse SDE** associated with forward SDE defined in the class `forward_sde`.



### **Mathematical Definition**
When the **forward SDE** is defined for $t \in (0,T]$ as:
\begin{equation*}
    \operatorname{d}\! \overrightarrow{X}_t = - \alpha(t) \overrightarrow{X}_t \operatorname{d}\! t+ \eta(t) \operatorname{d}\! B_t, \quad \overrightarrow{X}_0 \sim \pi
\end{equation*}
with:
- $ \forall t \in [0,T]$, $\overrightarrow{X}_t \in \mathbb{R}^d$.
- $\alpha: [0,T] \to [0, +\infty)$: drift coefficient.
- $\eta: [0,T] \to [0, +\infty)$: diffusion coefficient (sometimes called volatility).
- $(B_t)_{t \in [0,T]}$: standard Brownian motion of $\mathbb{R}^d$.

The **reverse SDE** is defined as:

$$
d\overleftarrow{X}_t = \left[ \alpha(T-t) \overleftarrow{X}_t + \eta^2(T-t)  \nabla \log p_{T-t}(\overleftarrow{X}_t) \right] \mathrm{d} t + \eta(T-t) \, \mathrm{d} B_t, 
$$
where:
- for $x \in \mathbb{R}^d$ and $t \in [0,T]$, $\nabla \log p_t(x) : \mathbb{R}^d \times [0,T] \to \mathbb{R}^d $ is the score function gradient of the log-likelihood or its parametric approximation $s_{\theta}(x,t)$.

The **Euler-Maruyama discretization** approximates this stochastic process by considering $N$ time intervals $t_1 =0 \leq \ldots \leq t_k \leq t_{k+1} \leq \ldots \leq t_N = T$ such that $T = \sum_{k = 1}^N \Delta_k$ with $\Delta_k = t_{k+1}-t_k$ so that

$$
\bar{X}_{k+1} = \bar{X}_k + \Delta_k \left[ \alpha(T-t_k) \bar{X}_k + \eta^2(T-t_k) \nabla \log p_{T-t_k}( \bar{X}_k) \right]  + \sqrt{\Delta_k} \eta(T-t_k)  Z_k,
$$

with:

- $Z_k \sim \mathcal{N}(0, I_d)$: an independent standard Gaussian noise.


### **Parameters**

| Parameter       | Type             | Description                                                                                       |
|-----------------|------------------|---------------------------------------------------------------------------------------------------|
| `init`         | `torch.Tensor`   | Initial state $\bar{X}_0$.                                |
| `sde`          | `object`         | SDE object with methods `alpha(t)` (drift), `eta(t)` (diffusion), and attribute `final_time`.     |
| `score_theta`  | `callable`       | Score function $\nabla \log p_t(x)$ (or its approximation $s_\theta(x, t)$).                       |
| `num_steps`    | `int`            | Number of time discretization steps $N$ (in this implementation the step-size is contant).                                                     |



### **Returns**

- `torch.Tensor`: Final state $\bar{X}_N$ approximating the solution to the reverse-time SDE.

---


## `EI_discr_sampler` Function Documentation

```python
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
```

The `EI_discr_sampler` function implements an "Exponential Integrator" (EI) discretization scheme for the **reverse-time SDE** that makes use of integrated quantities derived from the SDE as defined in the appendix of the paper.


### **Mathematical Definition**

When the **forward SDE** is defined for $t \in (0,T]$ as:
\begin{equation*}
    \operatorname{d}\! \overrightarrow{X}_t = - \alpha(t) \overrightarrow{X}_t \operatorname{d}\! t+ \eta(t) \operatorname{d}\! B_t,
\end{equation*}
where:
- for all $t \in [0,T]$, $\overrightarrow{X}_t \in \mathbb{R}^d$.
- $\alpha: [0,T] \to [0, +\infty)$: drift coefficient.
- $\eta: [0,T] \to [0, +\infty)$: diffusion coefficient (sometimes called volatility).
- $(B_t)_{t \in [0,T]}$: standard Brownian motion of $\mathbb{R}^d$.

The **reverse SDE** is defined as:
$$
d\overleftarrow{X}_t = \left[ \alpha(T-t) \overleftarrow{X}_t + \eta^2(T-t)  \nabla \log p_{T-t}(\overleftarrow{X}_t) \right] \mathrm{d} t + \eta(T-t) \, \mathrm{d} B_t, 
$$
where:
- for $x \in \mathbb{R}^d$ and $t \in [0,T]$, $\nabla \log p_t(x) : \mathbb{R}^d \times [0,T] \to \mathbb{R}^d $ is the score function gradient of the log-likelihood or its parametric approximation $s_{\theta}(x,t)$.

The **Exponential Integrator discretization** scheme approximates this stochastic process by considering $N$ time intervals $t_1 =0 \leq \ldots \leq t_k \leq t_{k+1} \leq \ldots \leq t_N = T$ such that $T = \sum_{k = 1}^N \Delta_k$ with $\Delta_k = t_{k+1}-t_k$ so that

$$
\bar{X}_{
k+1} = \mathrm{e}^{\int^{T-t_k}_{T-t_{k+1}} \alpha(t) \mathrm{d} t} \bar{X}_{k+1} + \nabla \log p_{T-t_k}( \bar{X}_k) * \mathrm{e}^{ - \int_0^{T-t_{k+1}} \alpha(t) \mathrm{d} t} \int_{T-t_{k+1}}^{T-t_k} \mathrm{e}^{\int_0^t \alpha (t)} \eta^2(t) \mathrm{d} t + \mathrm{e}^{ - \int_0^{T-t_{k+1}} \alpha(t) \mathrm{d} t} \sqrt{ \int_{T-t_{k+1}}^{T-t_k} \mathrm{e}^{2 \int_0^t \alpha (t)} \eta^2(t) \mathrm{d} t} Z_k
$$
where

- $Z_k \sim \mathcal{N}(0, I_d)$: an independent standard Gaussian noise.





### **Parameters**

| Parameter       | Type             | Description                                                                                 |
|-----------------|------------------|---------------------------------------------------------------------------------------------|
| `init`          | `torch.Tensor`   | Initial state $\bar{X}_0$.                              |
| `sde`           | `object`         | SDE object providing methods: `alpha_integrate(t)`, `I1(t)`, `I2(t)`, and attributes `final_time`, `device`. |
| `score_theta`   | `callable`       | Score function $\nabla \log p_t(x)$ (or its approximation $s_\theta(x, t)$).                 |
| `num_steps`     | `int`            | Number of discretization steps $N$ (in this implementation the step-size is contant).                                                     |

---

### **Returns**

- `torch.Tensor`: Final state $\bar{X}_N$ approximating the solution to the reverse-time SDE.

---

# `functions.py` : empirical processing & distances.

## Empirical Processing

### `normalize` Function Documentation

```python
def normalize(training_sample, rescale=1):
    means = torch.mean(training_sample, dim=0)
    std_devs = torch.std(training_sample, dim=0, unbiased=True)
    normalized_sample = (training_sample - means) / (rescale * std_devs)
    return normalized_sample, means, std_devs
```

### **Description**  
Standardizes each feature of the input data (`training_sample`) by subtracting its mean and dividing by its (scaled) standard deviation. 

### **Mathematical Definition**

Given a dataset $\mathcal{D} = \left( X_i \right)_{i =1}^n \in (\mathbb{R}^d)^n$, its mean vector $\mu$ and standard deviations $\sigma$ for each dimension:

$$
\mu_j = \frac{1}{n} \sum_{i=1}^n X_{ij}, 
\quad 
\sigma_j = \sqrt{\frac{1}{n-1} \sum_{i=1}^n (X_{ij} - \mu_j)^2},
$$

the normalized data is:

$$
X_{ij}^{(\mathrm{norm})} 
= \frac{X_{ij} - \mu_j}{\text{rescale} \times \sigma_j}.
$$

### **Parameters**

| Parameter          | Type             | Description                                                                    |
|--------------------|------------------|--------------------------------------------------------------------------------|
| `training_sample`  | `torch.Tensor`   | Input data of shape *(n, d)*, where *n* = number of samples and *d* = dimension. |
| `rescale`          | `float`          | Optional rescaling factor (default = 1). |

### **Returns**

- **`normalized_sample`** *(torch.Tensor)*: The standardized data of shape *(n, d)*.  
- **`means`** *(torch.Tensor)*: Vector of mean values for each dimension (shape *(d,)*).  
- **`std_devs`** *(torch.Tensor)*: Vector of standard deviations for each dimension (shape *(d,)*).

---


### `unnormalize` Function Documentation

```python
def unnormalize(normalized_sample, means, std_devs, rescale = 1):
    original_sample = (normalized_sample * rescale * std_devs) + means
    return original_sample
```

**Description:**  
Reverts the standardization process of the function `normalize`.

### **Parameters**

| Parameter           | Type             | Description                                                        |
|---------------------|------------------|--------------------------------------------------------------------|
| `normalized_sample` | `torch.Tensor`   | Normalized data of shape *(n, d)*.                                 |
| `means`             | `torch.Tensor`   | Vector of mean values of shape *(d,)*.                             |
| `std_devs`          | `torch.Tensor`   | Vector of standard deviations of shape *(d,)*.                     |
| `rescale`           | `float`          | Scaling factor used during normalization (default = 1.).            |

### **Returns**

- **`original_sample`** *(torch.Tensor)*: data recovered in the original scale.

---

### `empirical_mean_covar` Function Documentation

```python
def empirical_mean_covar(sample):
    mean = sample.mean(axis = 0)
    sample_centered = sample - mean
    covar = sample_centered.T @ sample_centered / (sample_centered.shape[0] - 1)
    return mean, covar
```

Computes the empirical mean and covariance matrix.

### **Parameters**

| Parameter  | Type             | Description                                                                      |
|------------|------------------|----------------------------------------------------------------------------------|
| `sample`   | `torch.Tensor`   | Data of shape *(n, d)*, where *n* = number of samples and *d* = dimension.       |

### **Returns**

- **`mean`** *(torch.Tensor)*: Vector of shape *(d,)* representing the empirical mean.  
- **`covar`** *(torch.Tensor)*: Matrix of shape *(d, d)* representing the empirical covariance.

---


## *Metrics & Distances*

### `relative_fisher_information_Gaussian` Function Documentation

```python
def relative_fisher_information_Gaussian(dataset, sde):
    d = dataset.d
    sigma_infty = sde.sigma_infty
    mean, covar = dataset.mean_covar()
    trace_covar = torch.trace(covar)
    norm_mean = torch.sum(mean**2)
    trace_inverse_covar = torch.trace(torch.inverse(covar))
    result = (1/sigma_infty**4) * (trace_covar + norm_mean) \
             - (2*d/sigma_infty**2) + trace_inverse_covar
    return result
```

Computes the **empirical relative Fisher Information** of a dataset of samples from Gaussian distribution $X_0 \sim \mathcal{N}(\mu_0, \Sigma_0)$ and the stationnary distribution of a VPSDE process (as defined in `forward_VPSDE`) $X_{\infty} \sim \mathcal{N}(0, \sigma^2 \mathrm{I}_d)$.

### **Mathematical Definition**
See Lemma E.4 in the Appendix of the paper.

### **Parameters**

| Parameter  | Type    | Description                                                      |
|------------|---------|------------------------------------------------------------------|
| `dataset`  | object  | An empirical object with a `mean_covar()` method returning *(mean, covar)*. |
| `sde`      | object  | VPSDE object with attribute `sigma_infty`       |

### **Returns**

- **`result`** *(float or tensor)*: the empirical relative Fisher Information between $X_0$ and $X_{\infty}$.

---

## `kl_divergence` Function Documentation

```python
def kl_divergence(mu1, sigma1, mu2, sigma2):
    d = len(mu1)
    delta_mu = mu2 - mu1
    inverse_sigma2 = torch.pinverse(sigma2)
    _, log_det_sigma1 = torch.linalg.slogdet(sigma1)
    _, log_det_sigma2 = torch.linalg.slogdet(sigma2)
    log_term = log_det_sigma2 - log_det_sigma1
    trace_term = torch.trace(inverse_sigma2 @ sigma1)
    delta_term = delta_mu @ inverse_sigma2 @ delta_mu[:, None]

    return 0.5 * (log_term - d + trace_term + delta_term).item()
```

### **Description:**  
Computes the **Kullback–Leibler divergence** between two multivariate Gaussian distributions: $\mathcal{N}(\mu_1, \Sigma_1)$ and $\mathcal{N}(\mu_2, \Sigma_2)$.

### **Mathematical Definition:**
See Lemma E.2 in the Appendix of the paper.


### **Parameters**
| Parameter   | Type             | Description                                           |
|-------------|------------------|-------------------------------------------------------|
| `mu1`       | `torch.Tensor`   | Mean of the first Gaussian, shape *(d,)*.            |
| `sigma1`    | `torch.Tensor`   | Covariance of the first Gaussian, shape *(d, d)*.    |
| `mu2`       | `torch.Tensor`   | Mean of the second Gaussian, shape *(d,)*.           |
| `sigma2`    | `torch.Tensor`   | Covariance of the second Gaussian, shape *(d, d)*.   |

### **Returns**

- **`kl_value`** *(float)*: $\mathrm{KL}\bigl(\mathcal{N}_1 \|\mathcal{N}_2\bigr)$.

---

## `wasserstein_w2` Function Documentation

```python
def wasserstein_w2(mu1, sigma1, mu2, sigma2):
    mu1_np = mu1.cpu().numpy()
    sigma1_np = sigma1.cpu().numpy()
    mu2_np = mu2.cpu().numpy()
    sigma2_np = sigma2.cpu().numpy()
    diff_term = np.sum((mu1_np - mu2_np)**2)
    sqrt_sigma1 = scipy.linalg.sqrtm(sigma1_np).real
    sqrt_last = scipy.linalg.sqrtm(sqrt_sigma1 @ sigma2_np @ sqrt_sigma1).real
    return math.sqrt((diff_term + np.trace(sigma1_np + sigma2_np - 2 * sqrt_last)).item())
```

**Description:**  
Computes the **2-Wasserstein** distance between two multivariate Gaussian distributions: $\mathcal{N}(\mu_1, \Sigma_1)$ and $\mathcal{N}(\mu_2, \Sigma_2)$.

### **Mathematical Definition**
See Lemma E.3 in the Appendix of the paper.


### **Parameters**

| Parameter   | Type             | Description                                            |
|-------------|------------------|--------------------------------------------------------|
| `mu1`       | `torch.Tensor`   | Mean of the first Gaussian, shape *(d,)*.             |
| `sigma1`    | `torch.Tensor`   | Covariance of the first Gaussian, shape *(d, d)*.     |
| `mu2`       | `torch.Tensor`   | Mean of the second Gaussian, shape *(d,)*.            |
| `sigma2`    | `torch.Tensor`   | Covariance of the second Gaussian, shape *(d, d)*.    |

### **Returns**

- **`w2_value`** *(float)*: $ \mathcal{W}_2(\mathcal{N}_1,\mathcal{N}_2)$.

---
