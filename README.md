# Bayesian Logistic Regression with $L_{\frac{1}{2}}$ prior

## Model setting

We use the data augmentation trick in logistic models proposed by https://arxiv.org/pdf/1205.0310.pdf.



Let $y_{i}$ be the number of successes, $n_{i}$ the number of trials and $x_{i}=(x_{i1},...,x_{ip})$ the vector of predictors for observation $i=1,2,3,...,N$. 

Let $y_{i} \sim \operatorname{Binom}\left(n_{i}, 1 /\left[1+e^{-\psi_{i}}\right]\right)$, where $\psi_{i}=x_{i}^{T} \beta$ is the log odds of success. 



The likelihood for observation $i$ is


$$
L_{i}(\beta)=\frac{\left[\exp \left(x_{i}^{T} \beta\right)\right]^{y_{i}}}{\left[1+\exp \left(x_{i}^{T} \beta\right)\right]^{n_{i}}} \propto \exp \left(\kappa_{i} x_{i}^{T} \beta\right) \int_{0}^{\infty} \exp \left[-\omega_{i}\left(x_{i}^{T} \beta\right)^{2} / 2\right] p\left(\omega_{i} | n_{i}, 0\right)dw_{i}
$$


where $\kappa_{i}=y_{i}-n_{i} / 2$ and $p\left(\omega_{i} | n_{i}, 0\right)$  is the density of a Polya–Gamma random variable with parameters $\left(n_{i}, 0\right)$. 



Then, conditional on $\omega$, we have a Gaussian likelihood of the form


$$
\prod_{i=1}^{N}L_{i}(\beta|\omega_{i})\propto \exp \left[-\frac{1}{2}(z-X \beta)^{T} \Omega(z-X \beta)\right]
$$


where $z=\left(\kappa_{1} / \omega_{1}, \ldots, \kappa_{N} / \omega_{N}\right)$ and $\Omega=\operatorname{diag}\left(\omega_{1}, \ldots, \omega_{N}\right)$. Let the  $L_{\frac{1}{2}}$ prior be assigned to $\beta$ .



## Partially Collapsed Gibbs Sampling

1. Sample $\beta \mid \lambda,\tau^{2},w  \sim N(\mu,\Sigma)$



where $\Sigma= (X^{T}DX+\lambda^{4}\Lambda^{-2})^{-1}$ , $\mu=\Sigma X^{T}D\kappa$ ,  $D=\mathrm{Diag}(\omega_{i})$ and $\Lambda=\mathrm{Diag}(\tau^{2})$



2. Sample $\lambda \mid \beta, b \sim \mathrm{Gamma}(2p+0.5,\sum_{j=1}^{p}|\beta_{j}|^{\frac{1}{2}}+1/b)$

   

3. Sample $b \mid \lambda \sim \mathrm{InvGamma}(1,1+\lambda)$

   

4. Sample $\frac{1}{v_{j}} \sim \operatorname{InvGaussian}\left(\sqrt{\frac{1}{4 \lambda^{ 2}\left|\beta_{j}\right|}}, \frac{1}{2}\right), \quad j=1, \ldots, p$

   

5. Sample $\frac{1}{\tau_{j}} \sim \operatorname{InvGaussian}\left(\frac{1}{\lambda^{2} v_{j}\left|\beta_{j}\right|}, \frac{1}{v_{j}}\right), \quad j=1, \ldots, p$

   

6. Sample $\omega_{i} \sim  \mathrm{PG}(n_{i},x_{i}^{T}\beta), \quad i=1,\dots,N$



## Coordinate descent with Proximal Newton map

In this section, we extend the coordinate descent algorithm from Gaussian likelihood to binary logistic likelihood. Our strategy is the same as https://www.jstatsoft.org/article/view/v033i01. By coding the response as $0/1$, the likelihood can be written as


$$
\begin{aligned}
\ell(\beta) &=\sum_{i=1}^{N}\left\{y_{i} \log p\left(x_{i} ; \beta\right)+\left(1-y_{i}\right) \log \left(1-p\left(x_{i} ; \beta\right)\right)\right\} \\
&=\sum_{i=1}^{N}\left\{y_{i} x_{i}\beta -\log \left(1+e^{ x_{i}\beta}\right)\right\}.
\end{aligned}
$$


If we apply the Newton algorithm to fit the (unpenalized) log-likelihood above, that is equivalent to using the iteratively reweighted least squares algorithm. For each iteration, we solve


$$
\beta^{\text {new }} \leftarrow \arg \min _{\beta}(\mathbf{z}-\mathbf{X} \beta)^{T} \mathbf{W}(\mathbf{z}-\mathbf{X} \beta)
$$



where $\mathbf{z}=\mathbf{X} \beta^{\text {old }}+\boldsymbol{W}^{-1}\left(Y-P(Y=1|X\beta^{old})\right)$ and 

$\boldsymbol{W}=\mathbf{diag}\left(P(Y=1|X\beta^{old})*(1-P(Y=1|X\beta^{old}))\right)$ is $N\times N$ diagonal matrix of weights.



For logistic regression with a non-separable Bridge penalty, the fitting algorithm consists of two steps. First, we create an outer loop, which computes the quadratic approximation of (unpenalized) log-likelihood as in (\ref{eq:likelihood}) based on the $\beta$ from last iteration. Then in the inner loop, we use coordinate descent to solve the penalized weighted least-squares problem


$$
\beta^{\text {new }} \leftarrow \arg \min _{\beta}(\mathbf{z}-\mathbf{X} \beta)^{T} \mathbf{W}(\mathbf{z}-\mathbf{X} \beta)+(2^{\gamma}p+0.5)\log\left(\sum_{j=1}^{p}|\beta_{j}|^{\frac{1}{2^{\gamma}}}+1/b\right).
$$



The solution to Equation (\ref{eq:Proximal}) in the inner loop is known as Proximal Newton Map https://arxiv.org/pdf/1206.1623.pdf. By slightly modifying the coordinate descent algorithm for penalized least square problem in https://arxiv.org/pdf/2108.03464.pdf,  we can get the coordinate descent algorithm for penalized weighted least square problem.



