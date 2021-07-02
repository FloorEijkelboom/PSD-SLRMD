import scipy.linalg as ln
import numpy as np
import copy 

class SubgradientDescent:
    def __init__(self, gamma, eta, max_iter=1000):
        self.eta = eta
        self.gamma = gamma
        self.max_iter = max_iter

    def objective_function(self, A, L, R):
        return self.gamma * ln.norm(A, 1) + (ln.norm(L, 'fro')**2 + ln.norm(R, 'fro')**2)

    def subgradient_A(self, A):
        Asign = np.sign(A)
        # replace the zero entries with num uniform [0, 1]
        Q = np.random.rand(len(A), len(A))
        Q[A!=0] = 0
        Q = (Q + Q.T) / 2
        if np.sum(Q) != 0:
            Q = Q/np.sum(Q)

        return self.gamma * (Asign + Q)
    
    def subgradient_L(self, L):
        return L.T
    
    def subgradient_R(self, R):
        return R

    def stepsize(self, subgradient):
        subgradient_length = ln.norm(subgradient, 'fro')
        if subgradient_length != 0:
            return self.eta / subgradient_length
        return 1

    def dykstra(self, A, L, R, C):
        A_p = (A + A.T)/2
        L_p = (C - (L + L.T)/2)*np.linalg.inv(R.T)
        R_p = R

        return A_p, L_p, R_p

    def average(A, L, R, C):
        A_p = (3 * A + A.T) / 4
        L_p = ((C - A)*np.linalg.inv(R.T) + L)/2
        R_p = R

        return A_p, L_p, R_p

    def gen_init_vals(self, C):
        U, Sigma, Vt = np.linalg.svd(C)
        A = np.zeros((len(C), len(C)))
        L, R = U * Sigma**(1/2), Vt.T * Sigma**(1/2)

        return A, L, R

    def predict(self, C, rank_gen):
        A_best, L_best, R_best = self.gen_init_vals(C)
        best_perf = self.objective_function(A_best, L_best, R_best)

        for i in range(self.max_iter):
            A_cur, L_cur, R_cur = copy.deepcopy(A_best), copy.deepcopy(L_best), copy.deepcopy(R_best)

            A_cur -= self.stepsize(A_cur) * self.subgradient_A(A_cur)
            L_cur -= self.stepsize(L_cur) * self.subgradient_L(L_cur)
            R_cur -= self.stepsize(R_cur) * self.subgradient_R(R_cur)
            
            A_cur, L_cur, R_cur = self.dykstra(A_cur, L_cur, R_cur, C)
            cur_perf = self.objective_function(A_cur, L_cur, R_cur)

            if cur_perf <= best_perf:
                A_best = A_cur
                L_best = L_cur
                R_best = R_cur
                best_perf = cur_perf
            
            if np.linalg.matrix_rank(L_best * R_best.T) <= rank_gen:
                return A_best, L_best*R_best.T, best_perf
        
        return A_best, L_best*R_best.T, best_perf

class ADM_S:
    def __init__(self, gamma, xi, delta, max_iter=500):
        self.gamma = gamma
        self.xi = xi
        self.delta = delta
        self.max_iter = max_iter

    def objective_function(self, A, L, R, C, Lambda, Kappa):
        value = self.gamma * ln.norm(A, 1)
        value += (ln.norm(L, 'fro')**2 + ln.norm(R.T, 'fro')**2) / 2
        value += np.trace((C - L * R.T - A).T * Lambda)
        value += np.trace((A - A.T).T * Kappa)
        return value + (self.xi/2) * (ln.norm(A + L * R.T - C, 'fro')**2  + ln.norm(A - A.T, 'fro')**2)

    def update_L(self, A, R, C, Lambda):
        return (-Lambda - self.xi*(A - C))*R*np.linalg.inv(np.eye(len(C)) + self.xi * R.T * R)

    def update_R(self, A, L, R, C, Lambda):
        return (-Lambda - self.xi * (A - C)).T * L * np.linalg.inv(np.eye(len(C)) + self.xi * L.T * L)

    def update_A(self, A, L, R, C, Lambda, Kappa):
        M = C - L * R.T + (Lambda + Kappa - Kappa.T)/self.xi
        M -= np.ones(shape=(len(C), len(C))) * (self.gamma/self.xi)
        M[M < 0] = 0
        return M
    
    def gen_init_vals(self, C):
        U, Sigma, Vt = np.linalg.svd(C)
        A = np.zeros((len(C), len(C)))
        L, R = U * Sigma**(1/2), Vt.T * Sigma**(1/2)
        Lambda, Kappa = np.zeros((len(C), len(C))), np.zeros((len(C), len(C)))

        return A, L, R, Lambda, Kappa

    def predict(self, C, rank_gen):
        A, L, R, Lambda, Kappa = self.gen_init_vals(C)

        for i in range(self.max_iter):
            A = self.update_A(A, L, R, C, Lambda, Kappa)
            L = self.update_L(A, R, C, Lambda)
            R = self.update_R(A, L, R, C, Lambda)

            Lambda -= self.delta * (C - L * R.T - A)
            Kappa -= self.delta * (A - A.T)


            if np.linalg.matrix_rank(L*R.T) <= rank_gen:
                return A, L * R.T, i
            
        return A, L * R.T, i

class ADM_PSD:
    def __init__(self, gamma, xi, delta, max_iter=500):
        self.gamma = gamma
        self.xi = xi
        self.delta = delta
        self.max_iter = max_iter

    def objective_function(self, A, L, R, C, Lambda, Mu):
        value = self.gamma * ln.norm(A, 1)
        value += (ln.norm(L, 'fro')**2 + ln.norm(R.T, 'fro')**2) / 2
        value += np.trace((C - L * R.T - A).T * Lambda)
        value += np.trace((L - R).T * Mu)
        return value + (self.xi/2) * (ln.norm(A + L * R.T - C, 'fro')**2  + ln.norm(A - A.T, 'fro')**2)

    def update_L(self, A, R, C, Lambda, Mu):
        return (-Mu - (Lambda + self.xi*(A - C + np.eye(len(C)))*R))*np.linalg.inv(np.eye(len(C)) + self.xi * (R.T * R - np.eye(len(C))))

    def update_R(self, A, L, C, Lambda, Mu):
        return (Mu - (Lambda.T + self.xi*(A - C - np.eye(len(C))).T*L))*np.linalg.inv(np.eye(len(C)) + self.xi * (L.T * L + np.eye(len(C))))

    def update_A(self, L, R, C, Lambda):
        M = C - L * R.T + Lambda/self.xi
        M -= np.ones(shape=(len(C), len(C))) * (self.gamma/self.xi)
        M[M < 0] = 0
        return M
    
    def gen_init_vals(self, C):
        U, Sigma, Vt = np.linalg.svd(C)
        A = np.zeros((len(C), len(C)))
        L, R = U * Sigma**(1/2), Vt.T * Sigma**(1/2)
        Lambda, Mu = np.zeros((len(C), len(C))), np.zeros((len(C), len(C)))

        return A, L, R, Lambda, Mu

    def predict(self, C, rank_gen):
        A, L, R, Lambda, Mu = self.gen_init_vals(C)

        for i in range(self.max_iter):
            A = self.update_A(L, R, C, Lambda)
            L = self.update_L(A, R, C, Lambda, Mu)
            R = self.update_R(A, L, C, Lambda, Mu)

            Lambda -= self.delta * (C - L * R.T - A)
            Mu -= self.delta * (L - R)
            
            if np.linalg.matrix_rank(L*R.T) <= rank_gen:
                return A, L * R.T, i
            
        return A, L*R.T, i

class ADM:
    def __init__(self, gamma, xi, delta, max_iter=500):
        self.gamma = gamma
        self.xi = xi
        self.delta = delta
        self.max_iter = max_iter

    def objective_function(self, A, L, R, C, Lambda):
        value = self.gamma * ln.norm(A, 1)
        value += (ln.norm(L, 'fro')**2 + ln.norm(R.T, 'fro')**2) / 2
        value += np.trace((C - L * R.T - A).T * Lambda)
        return value + (self.xi/2) * (ln.norm(A + L * R.T - C, 'fro')**2)

    def update_L(self, A, R, C, Lambda):
        return self.xi * (C - A - Lambda) * R * np.linalg.inv(np.eye(len(C)) + self.xi * R.T * R)

    def update_R(self, A, L, R, C, Lambda):
        return self.xi * (C - A - Lambda).T * L * np.linalg.inv(np.eye(len(C)) + self.xi *L.T * L)

    def update_A(self, A, L, R, C, Lambda):
        M = C - L * R.T - Lambda/self.xi
        M -= np.ones(shape=(len(C), len(C))) * (self.gamma/self.xi)
        M[M < 0] = 0
        return M
    
    def gen_init_vals(self, C):
        U, Sigma, Vt = np.linalg.svd(C)
        A = np.zeros((len(C), len(C)))
        L, R = U * Sigma**(1/2), Vt.T * Sigma**(1/2)
        Lambda = np.zeros((len(C), len(C)))

        return A, L, R, Lambda

    def predict(self, C, rank_gen):
        A, L, R, Lambda = self.gen_init_vals(C)

        for i in range(self.max_iter):
            A = self.update_A(A, L, R, C, Lambda)
            L = self.update_L(A, R, C, Lambda)
            R = self.update_R(A, L, R, C, Lambda)

            Lambda -= self.delta * (C - L * R.T - A)

            if np.linalg.matrix_rank(L*R.T) <= rank_gen:
                return A, L * R.T, i
            
        return A, L * R.T, i