# Bayesian Logistic Regression with $L_{\frac{1}{2}}$ prior

## Model setting

We use the data augmentation trick in logistic models proposed by https://arxiv.org/pdf/1205.0310.pdf.



Let $y_{i}$ be the number of successes, $n_{i}$ the number of trials and $x_{i}=(x_{i1},...,x_{ip})$ the vector of predictors for observation $i\in\left\{1,2,3,...,N\right\}$. 

Let $y_{i} \sim \operatorname{Binom}\left(n_{i}, 1 /\left\{1+e^{-\psi_{i}}\right\}\right)$, where $\psi_{i}=x_{i}^{T} \beta$ is the log odds of success. 



The likelihood for observation $i$ is


$$
L_{i}(\beta)=\frac{\left\{\exp \left(x_{i}^{T} \beta\right)\right\}^{y_{i}}}{\left\{1+\exp \left(x_{i}^{T} \beta\right)\right\}^{n_{i}}} \propto \exp \left(\kappa_{i} x_{i}^{T} \beta\right) \int_{0}^{\infty} \exp \left\{-\omega_{i}\left(x_{i}^{T} \beta\right)^{2} / 2\right\} p\left(\omega_{i} | n_{i}, 0\right)dw_{i}
$$


where $\kappa_{i}=y_{i}-n_{i} / 2$ and $p\left(\omega_{i} | n_{i}, 0\right)$  is the density of a Polya–Gamma random variable with parameters $\left(n_{i}, 0\right)$. 



Then, conditional on $\omega$, we have a Gaussian likelihood of the form


$$
\prod_{i=1}^{N}L_{i}(\beta|\omega_{i})\propto \exp \left\{-\frac{1}{2}(z-X \beta)^{T} \Omega(z-X \beta)\right\}
$$


where $z=\left(\kappa_{1} / \omega_{1}, \ldots, \kappa_{N} / \omega_{N}\right)$ and $\Omega=\operatorname{diag}\left(\omega_{1}, \ldots, \omega_{N}\right)$. Let the  $L_{\frac{1}{2}}$ prior be assigned to $\beta$ .



## PCG sampler

