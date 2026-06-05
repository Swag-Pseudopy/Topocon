import warnings
import time
import numpy as np
import cvxpy as cp
import networkx as nx
import matplotlib.pyplot as plt
from ripser import ripser
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
 
warnings.filterwarnings("ignore")
 
class TopoCon:
 
    def __init__(self, n_neighbors: int = 15,
                 nu: float = 0.1, gamma: float = 1e-3,
                 fusion_thresh: float = 0.1, tau_k: int = 5):
        self.l              = n_neighbors   # neighbourhood size
        self.nu             = nu            # fusion strength
        self.gamma          = gamma         # shrinkage regulariser
        self.fusion_thresh  = fusion_thresh # sparsity threshold for W
        self.tau_k          = tau_k         # k for adaptive tau estimation
 
    def _compute_persistence(self, X: np.ndarray) -> np.ndarray:
  
        n = X.shape[0]
        nn = NearestNeighbors(n_neighbors=self.l).fit(X)
        _, indices = nn.kneighbors(X)
 
        vectors = []
        for i in range(n):
            dgms = ripser(X[indices[i]], maxdim=1)["dgms"]
 
            # H0 lifetimes 
            h0 = dgms[0][:, 1] - dgms[0][:, 0]
            h0 = h0[np.isfinite(h0)]
 
            # H1 lifetimes 
            h1 = dgms[1][:, 1] - dgms[1][:, 0]
            h1 = h1[np.isfinite(h1)]
 
            vectors.append(np.concatenate([h0, h1]))
 
        # Pad to uniform length M and pack into column matrix
        M = max(len(v) for v in vectors)
        P = np.zeros((M, n))
        for i, v in enumerate(vectors):
            P[: len(v), i] = v
 
        return P
 
    def _build_similarity(P: np.ndarray) -> np.ndarray:
        dist = pairwise_distances(P.T)            # (n, n)
        sigma = np.median(dist)
        W = np.exp(-(dist ** 2) / (2 * sigma ** 2))
        return W
 
    
    def _convex_clustering(self, P: np.ndarray, W: np.ndarray) -> np.ndarray:
    
        M, n = P.shape
        U = cp.Variable((M, n))
 
        objective = 0.5 * cp.norm(P - U, "fro") ** 2 \
                  + (self.gamma / 2) * cp.norm(U, "fro") ** 2
 
        # Fusion terms (sparse: only edges above threshold)
        fusion = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                if W[i, j] > self.fusion_thresh:
                    fusion = fusion + W[i, j] * cp.norm(U[:, i] - U[:, j], 2)
 
        problem = cp.Problem(cp.Minimize(objective + self.nu * fusion))
        problem.solve(solver=cp.SCS, verbose=False)
 
        if U.value is None:
            raise RuntimeError("CVXPY solver did not converge.")
 
        return U.value  # (M, n)
 
    def _estimate_tau(self, U: np.ndarray) -> float:
        nbrs = NearestNeighbors(n_neighbors=self.tau_k + 1).fit(U.T)
        dists, _ = nbrs.kneighbors(U.T)
        return float(np.median(dists[:, -1]))
 
    def _cluster(self, U: np.ndarray) -> np.ndarray:
        n   = U.shape[1]
        tau = self._estimate_tau(U)
 
        Ut   = U.T
        nbrs = NearestNeighbors(radius=tau).fit(Ut)
 
        G = nx.Graph()
        G.add_nodes_from(range(n))
 
        for i in range(n):
            neighbours = nbrs.radius_neighbors([Ut[i]], return_distance=False)[0]
            for j in neighbours:
                if i < j:
                    G.add_edge(i, j)
 
        labels = np.zeros(n, dtype=int)
        for cluster_id, component in enumerate(nx.connected_components(G)):
            for node in component:
                labels[node] = cluster_id
 
        self.tau_ = tau     
        return labels
 
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
      
        # Step 1–3: persistence matrix P
        P = self._compute_persistence(X)
 
        # Step 4: similarity matrix W
        W = self._build_similarity(P)
 
        # Step 5: convex clustering
        U = self._convex_clustering(P, W)
 
        # Step 6: final labels
        return self._cluster(U)
