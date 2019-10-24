# BayessianCG
### Realization of Bayessian conjugate gradient method from https://arxiv.org/pdf/1801.05242.pdf
### Project proposal is placed here: https://drive.google.com/file/d/1Q3eGJIWKYQLPy2m6AHsE7H7L_--BpPhJ/view?usp=sharing
### The goal of project is to implement BCG and investigate its properties in terms of point estimation, posterior covariance matrix and uncertianty quantification. Also we propose two real problems those can be solved by BCG
### Our team: Andrey Savinov, Daria Riabukhina, Lusine Airapetyan.
### All funсtions are in BCG_CG_ichol.py. Just clone repository and run following notebooks to reproduce our results: Point_Trace_UQ_est.ipynb and Poisson.ipynb


## Results

1. Convergence in mean of BayesCG (BCG). For several independent test problems, the error was computed. The standard CG method was compared to variants of BayesCG, corresponding to different prior covariances. The search directions used for BayesCG were either computed sequentially.

![Point estimation](https://github.com/AndreySavinov/BayessianCG/blob/master/images/point_est.png)


2. Convergence in posterior covariance of BCG measured by tr(Σm)/tr(Σ0).

![Trace estimation](https://github.com/AndreySavinov/BayessianCG/blob/master/images/traces_est.png)


3. Assessment of the uncertainty quantification provided by the Gaussian BayesCG method, with different choices of Σ0.

![Uncertainty Quantification](https://github.com/AndreySavinov/BayessianCG/blob/master/images/UQ1.png)

4. Convergence of the posterior for a linear system arising from a discretisation of the Poisson PDE for different choice of prior (left) and function reconstruction (right).

![Poisson](https://github.com/AndreySavinov/BayessianCG/blob/master/images/Poisson.png)
