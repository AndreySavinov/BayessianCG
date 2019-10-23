import torch
import numpy as np

class BCG():
    
    def __init__(self, A, b, prior_mean, prior_cov, eps, m_max, batch_directions=False):
        
        self.A = A
        self.b = b
        self.x0 = prior_mean
        self.sigma0 = prior_cov
        self.eps = eps
        self.max = m_max
        self.batch_directions = batch_directions
        
    def bcg(self, x_true):
               
        A = self.A
        b = self.b
        x0 = self.x0
        sigma0 = self.sigma0
        eps = self.eps
        m_max = self.max
        batch_directions = self.batch_directions
        
        r_m = b - A.dot(x0)
        r_m_dot_r_m = r_m.T.dot(r_m)
        s_m = r_m
        x_m = x0
        
        sigmaF = np.zeros(sigma0.shape)
        search_directions = np.zeros(sigma0.shape)
        
        nu_m = 0
        m = 0
        d = b.shape[0]
        rel_error = np.zeros(m_max)
        rel_trace = np.zeros(m_max)
        
        while True:
            
            sigma_At_s = np.dot(sigma0, np.dot(A.T, s_m))
            A_sigma_A_s = np.dot(A, sigma_At_s)
            
            E_2 = np.dot(s_m.T, A_sigma_A_s)
            alpha_m = r_m_dot_r_m / E_2
            x_m += alpha_m * sigma_At_s
            r_m -= alpha_m * A_sigma_A_s
            
            if batch_directions:
                search_directions[:, m] = s_m.reshape(100,)/np.sqrt((s_m.T.dot(A.dot(sigma0.dot(A.T.dot(s_m))))))

            
            rel_error[m] = np.linalg.norm(x_true-x_m)/np.linalg.norm(x_true)
            
            nu_m += r_m_dot_r_m * r_m_dot_r_m / E_2
            sigma_m = np.sqrt((d - 1 - m) * nu_m / (m + 1)) ##??
            prev_r_m_dot_r_m = r_m_dot_r_m
            r_m_dot_r_m = np.dot(r_m.T, r_m)
            E = np.sqrt(E_2)
            
            sigmaF[:, m] = (sigma_At_s/E).reshape(d,)
                       
            m +=1
            
            Sigma_m = sigma0 - sigmaF[:, :m].dot(sigmaF[:, :m].T)
            rel_trace[m-1] = np.trace(Sigma_m)/np.trace(sigma0)
            
            if batch_directions:
                s_m = r_m
                for i in range(m):
                    coeff = r_m.T.dot(A.dot(sigma0.dot(A.T.dot(search_directions[:, i].reshape(d,1)))))                    
                    s_m -= coeff*search_directions[:, i].reshape(d,1)
            else:
                beta_m = r_m_dot_r_m / prev_r_m_dot_r_m
                s_m = r_m + beta_m *s_m
            
            
            if sigma_m < eps:
                break
                
            if m == m_max or m == d:
                break
                
        return x_m, Sigma_m, nu_m/m, rel_error, rel_trace

    
def conjugate_grad(A, b, maxiter, x_true):
    n = len(b)
    x = np.zeros([n, 1])
    sigmaF = np.zeros(A.shape)
    rel_error = np.zeros(n)
    trace_error = np.zeros(n)
    r = np.dot(A, x) - b
    d = b.shape[0]
    s = - r
    r_k_norm = np.dot(r.T, r)
    for i in range(maxiter):
        As = np.dot(A, s)
        alpha = r_k_norm / np.dot(s.T, As)
        x += alpha * s
        
        rel_error[i] = np.linalg.norm(x_true-x)/np.linalg.norm(x_true)
        sigmaF[:, i] = s.reshape(d,)/np.sqrt(np.dot(s.T, As))
        Sigma_m = np.eye(d) - sigmaF[:, :i+1].dot(sigmaF[:, :i+1].T)
        trace_error[i] = np.trace(Sigma_m)/np.trace(np.eye(d))
        
        r += alpha * As
        r_kplus1_norm = np.dot(r.T, r)
        beta = r_kplus1_norm / r_k_norm
        r_k_norm = r_kplus1_norm
        
        s = beta * s - r
    return x, rel_error, trace_error


def ichol(A):
    n = A.shape[0]
    L = np.zeros(A.shape)
    for i in range(n):
        sqrt_diag = A[i,i]
        for k in range(i):
            tmp = L[i,k]
            sqrt_diag -= tmp*tmp
        sqrt_diag = np.sqrt(sqrt_diag)
        L[i,i] = sqrt_diag
        sqrt_diag = 1./sqrt_diag
        for j in range(i+1, n): 
            tmp = A[j,i]
            if tmp == 0:
                continue
            for k in range(i):
                tmp -= L[i,k]*L[j,k]

            tmp *= sqrt_diag
            L[j,i] = tmp
            
    return L        