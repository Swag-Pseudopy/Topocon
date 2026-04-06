import numpy as np
import cvxpy as cp

class RCC:
    def __init__(self, alpha=0.1, beta=0.1, max_iter=10):
        self.alpha = alpha # Regularization for clusters [cite: 2251]
        self.beta = beta   # Row-sparsity for outliers [cite: 2264]
        self.max_iter = max_iter

    def fit_predict(self, X, n_clusters=2):
        d, n = X.T.shape # [cite: 2248]
        P = np.zeros((d, n))
        Q = np.zeros((d, n))
        X_mat = X.T

        for _ in range(self.max_iter):
            # Update P (Convex Clustering on purified data) [cite: 2294, 2296]
            U_purified = X_mat - Q
            P_var = cp.Variable((d, n))
            # Simplified CC update for speed
            obj_p = 0.5 * cp.norm(U_purified - P_var, "fro")**2
            prob_p = cp.Problem(cp.Minimize(obj_p)) 
            prob_p.solve()
            P = P_var.value

            # Update Q (Proximal projection) [cite: 2300, 2301]
            diff = X_mat - P
            for i in range(d):
                row_norm = np.linalg.norm(diff[i, :])
                Q[i, :] = max(0, 1 - self.beta/row_norm) * diff[i, :] if row_norm > 0 else 0
        
        return AgglomerativeClustering(n_clusters=n_clusters).fit_predict(P.T)
