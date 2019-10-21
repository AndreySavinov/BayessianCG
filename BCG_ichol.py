import torch

class BCG():
    
    def __init__(self, A, b, prior_mean, prior_cov, eps, m_max, batch_directions, detailed = True):
        
        self.A = A
        self.b = b
        self.x0 = prior_mean
        self.sigma0 = prior_cov
        self.eps = eps
        self.max = m_max
        self.detailed = detailed
        self.batch_directions = batch_directions
        
    def bcg(self):
        
        sigmaF = [] #np.concatenate?
        x_m_det = []
        search_directions = []
        A_sigma_A_search_directions = []
        search_normalisations = []
        detailed = self.detailed
        batch_directions=self.batch_directions
        A = self.A
        b = self.b
        x0 = self.x0
        sigma0 = self.sigma0
        eps = self.eps
        m_max = self.max
        
        r_m = b - torch.mm(A, x0)
        r_m_dot_r_m = torch.mm(r_m.t(), r_m)
        s_m = r_m
        x_m = x0
        
        nu_m = 0
        m = 0
        d = b.shape[0]
        
        while True:
            sigma_At_s = torch.mm(sigma0, torch.mm(A.t(), s_m))
            A_sigma_A_s = torch.mm(A, sigma_At_s)
            
            E_2 = torch.mm(s_m.t(), A_sigma_A_s)
            alpha_m = r_m_dot_r_m / E_2
            x_m += alpha_m * sigma_At_s
            r_m -= alpha_m * A_sigma_A_s
            nu_m += r_m_dot_r_m * r_m_dot_r_m / E_2
            sigma_m = ((d - 1 - m) * nu_m / (m + 1)).sqrt() ##??
            prev_r_m_dot_r_m = r_m_dot_r_m
            r_m_dot_r_m = torch.mm(r_m.t(), r_m)
            E = E_2.sqrt()
            sigmaF.append(sigma_At_s / E)
            
            if detailed or batch_directions:
                x_m_det.append(x_m)
                search_directions.append(s_m)
                A_sigma_A_search_directions.append(A_sigma_A_s)
                search_normalisations.append(E_2)
            
            
            m +=1
            
            if batch_directions:
                s_m = r_m
                for i in range(m):
                    coeff = r_m.t().mm(A_sigma_A_search_directions[i])/search_normalisations[i]
                    print(coeff)
                    s_m -= coeff*search_directions[i]                
            else:            
                beta_m = r_m_dot_r_m / prev_r_m_dot_r_m
                s_xm = r_m + beta_m *s_m
            
            #add minimal no of iterations
            if sigma_m < eps:
                break
            '''else sqrt(r_m_dot_r_m) < eps: - traditional residual-minimising strategy
                break'''
            if m == m_max or m == d:
                break
                
        if detailed:
            return x_m_det, sigmaF, nu_m/m, search_directions, search_normalisations
        else:
            return x_m, sigmaF, nu_m/m

def ichol(A):
    n = A.shape[0]
    L = torch.zeros_like(A)
    for i in range(n):
        sqrt_diag = A[i,i]
        for k in range(i):
            tmp = L[i,k]
            sqrt_diag -= tmp*tmp
        sqrt_diag = torch.sqrt(sqrt_diag)
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