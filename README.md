# BayessianCG
### Realization of Bayessian conjugate gradient method from https://arxiv.org/pdf/1801.05242.pdf
### Project proposal is placed here: https://drive.google.com/file/d/1Q3eGJIWKYQLPy2m6AHsE7H7L_--BpPhJ/view?usp=sharing
### The goal of project is to implement BCG and investigate its properties in terms of point estimation, posterior covariance matrix and uncertianty quantification. Also we propose two real problems those can be solved by BCG
### Our team: Andrey Savinov, Daria Riabukhina, Lusine Airapetyan.
### All funсtions are in BCG_CG_ichol.py. Just clone repository and run following notebooks to reproduce our results: Point_Trace_UQ_est.ipynb and Poisson.ipynb


## Results

Convergence in mean of BayesCG (BCG). For several independent test problems, x ∗ ∼ μ ref , the error kx m − x ∗ k 2 was computed. The standard CG method was compared to variants of BayesCG, corresponding to different prior covariances. The search directions used for BayesCG were either computed sequentially.

![Point estimation](https://github.com/AndreySavinov/BayessianCG/images/point_est.png)
