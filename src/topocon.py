import numpy as np
import cvxpy as cp
from ripser import ripser
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances

class TopoCon:
    def __init__(self, n_neighbors=15, n_clusters=2, nu=0.1, gamma=1e-3):
        self.l = n_neighbors # [cite: 1863, 1900]
        self.c = n_clusters  # [cite: 1906]
        self.nu = nu         # Fusion strength [cite: 1921]
        self.gamma = gamma   # Shrinkage [cite: 1923]

    def fit_predict(self, X):
        n = X.shape[0]
        # Step 1 & 2: Local VR Filtration [cite: 1867, 1871]
        nn = NearestNeighbors(n_neighbors=self.l).fit(X)
        _, indices = nn.kneighbors(X)
        
        persistence_vectors = []
        for i in range(n):
            dgms = ripser(X[indices[i]], maxdim=1)['dgms']
            # Lifetimes for H0 and H1 [cite: 1872, 1884]
            h0 = dgms[0][:, 1] - dgms[0][:, 0]
            h1 = dgms[1][:, 1] - dgms[1][:, 0]
            persistence_vectors.append(np.concatenate([h0[np.isfinite(h0)], h1]))

        # Step 3: Persistence Matrix P [cite: 1890, 1903]
        M = max(len(v) for v in persistence_vectors)
        P = np.zeros((M, n))
        for i, v in enumerate(persistence_vectors):
            P[:len(v), i] = v

        # Step 4: Similarity Matrix W [cite: 1914, 1916]
        dist_P = pairwise_distances(P.T)
        sigma = np.median(dist_P)
        W = np.exp(-(dist_P**2) / (2 * sigma**2))

        # Step 5: Convex Optimization [cite: 1919]
        U = cp.Variable((M, n))
        objective = 0.5 * cp.norm(P - U, "fro")**2
        
        fusion = 0
        for i in range(n):
            for j in range(i + 1, n):
                if W[i, j] > 0.1: # Threshold for speed
                    fusion += W[i, j] * cp.norm(U[:, i] - U[:, j], 2)
        
        prob = cp.Problem(cp.Minimize(objective + self.nu * fusion + (self.gamma/2)*cp.norm(U, "fro")**2))
        prob.solve(solver=cp.SCS)

        # Step 6: Final HAC [cite: 1926, 1929]
        return AgglomerativeClustering(n_clusters=self.c).fit_predict(U.value.T)
