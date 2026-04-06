import numpy as np

class BCC:
    def __init__(self, gamma=1.0, lam=0.1, max_iter=20):
        self.gamma = gamma # [cite: 2691]
        self.lam = lam     # [cite: 2695]
        self.max_iter = max_iter

    def fit_predict(self, X, n_clusters=2):
        n, p = X.shape
        mu = X.copy()
        w = np.full(p, 1.0/p) # [cite: 2764]
        
        for _ in range(self.max_iter):
            # Update mu (Centroids) [cite: 2754]
            # Parallel component-wise update simplified
            for l in range(p):
                weight_term = w[l]**2 + self.lam * w[l]
                mu[:, l] = (weight_term * X[:, l]) / (weight_term + self.gamma)

            # Update w (Feature Weights via Bisection) [cite: 2758, 2764]
            D = np.sum((X - mu)**2, axis=0)
            # Root finding for alpha* such that sum(S(alpha/D, lam)) = 2
            w = np.maximum(0, 1/D - self.lam) # Simplified update
            w /= np.sum(w)

        return AgglomerativeClustering(n_clusters=n_clusters).fit_predict(mu)
