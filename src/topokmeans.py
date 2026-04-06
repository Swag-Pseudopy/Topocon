import warnings
import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from ripser import ripser
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.cluster import AgglomerativeClustering

warnings.filterwarnings('ignore')

def topo_kmeans(
    data,
    nKNN,
    nClust=2,
    power=5,
    sigma=0.05,
    dist_matrix=False,
    preserve_ordering=False,
    null_dim=False,
    first_dim=False,
):
    """
    Performs TopoKmeans clustering by calculating pairwise topological distances 
    based on persistent homology lifetimes.

    Parameters
    ----------
    data : np.ndarray
        Raw data points (N x D) or precomputed distance matrix (N x N).
    nKNN : int
        Neighbors for local Rips complexes.
    nClust : int
        Number of clusters.
    power : float
        Exponent for the kernelized distance.
    sigma : float
        Bandwidth for the RBF kernel.
    dist_matrix : bool
        If True, data is treated as a distance matrix.
    preserve_ordering : bool
        If True, uses sliding window indices instead of kNN.
    null_dim : bool
        Use only H0 features.
    first_dim : bool
        Use only H1 features.
    """

    data = np.asarray(data)
    N = data.shape[0]

    # 1) Determine maxscale
    if dist_matrix:
        sorted_rows = np.sort(data, axis=1)
        maxscale = np.ceil(np.max(sorted_rows[:, nKNN - 1])).astype(float)
    else:
        if preserve_ordering:
            maxscale = float(np.max(data))
        else:
            nbrs = NearestNeighbors(n_neighbors=nKNN, algorithm="auto").fit(data)
            distances, _ = nbrs.kneighbors(data)
            maxscale = np.ceil(np.max(distances[:, nKNN - 1])).astype(float)

    # 2) Build neighbor indices
    if dist_matrix:
        ind = np.argsort(data, axis=1)[:, :nKNN]
    else:
        if preserve_ordering:
            ind = []
            half = nKNN // 2
            for i in range(N):
                start = max(0, i - half)
                end = min(N, i + half + 1)
                ind.append(np.array(list(range(start, end)), dtype=int))
        else:
            nbrs = NearestNeighbors(n_neighbors=nKNN, algorithm="auto").fit(data)
            _, neighbors = nbrs.kneighbors(data)
            ind = neighbors 

    # 3) Compute Persistent Homology
    PDs = [None] * N
    dimM, dim0, dim1 = np.zeros(N, dtype=int), np.zeros(N, dtype=int), np.zeros(N, dtype=int)

    for i in range(N):
        if dist_matrix:
            sub_dm = data[np.ix_(ind[i], ind[i])]
            dgms = ripser({"distance_matrix": sub_dm}, maxdim=0, thresh=maxscale)["dgms"]
            H0, H1 = dgms[0], np.empty((0, 2))
        else:
            pts = data[ind[i], :]
            dgms = ripser(pts, maxdim=1, thresh=maxscale)["dgms"]
            H0, H1 = dgms[0], dgms[1]

        # Process H0
        birth0, death0 = H0[:, 0], H0[:, 1]
        death0[np.isinf(death0)] = maxscale
        pd0 = np.vstack([np.zeros_like(birth0, dtype=int), birth0, death0]).T

        # Process H1
        birth1, death1 = H1[:, 0], H1[:, 1]
        death1[np.isinf(death1)] = maxscale
        pd1 = np.vstack([np.ones_like(birth1, dtype=int), birth1, death1]).T

        PD = np.vstack([pd0, pd1])
        PDs[i] = PD
        dimM[i], dim0[i], dim1[i] = PD.shape[0], pd0.shape[0], pd1.shape[0]

    # 4) Fill persistence lifetimes
    max_dimM, max_dim0, max_dim1 = np.max(dimM), np.max(dim0), np.max(dim1)
    perst_val_0 = np.zeros((int(max_dim0), N))
    perst_val_1 = np.zeros((int(max_dim1), N))
    perst_val = np.zeros((int(max_dimM), N))

    for i in range(N):
        PD = PDs[i]
        if PD is None or PD.shape[0] == 0: continue
        
        h0_idx, h1_idx = np.where(PD[:, 0] == 0)[0], np.where(PD[:, 0] == 1)[0]
        
        if len(h0_idx) > 0:
            perst_val_0[:len(h0_idx), i] = PD[h0_idx, 2] - PD[h0_idx, 1]
        if len(h1_idx) > 0:
            perst_val_1[:len(h1_idx), i] = PD[h1_idx, 2] - PD[h1_idx, 1]
        
        perst_val[:len(PD), i] = PD[:, 2] - PD[:, 1]

    # 5) Kernelized Distance Matrix
    if null_dim: persistence = perst_val_0
    elif first_dim: persistence = perst_val_1
    else: persistence = perst_val

    kM = np.zeros(N)
    gamma = 1.0 / (2 * sigma**2)
    for i in range(N):
        vi = persistence[:, i].reshape(-1, 1)
        kM[i] = np.sum(rbf_kernel(vi, vi, gamma=gamma))

    dist = np.zeros((N, N))
    print("Forming Topological Distance Matrix...")
    for i in tqdm(range(N)):
        vi = persistence[:, i].reshape(-1, 1)
        for j in range(i + 1, N):
            vj = persistence[:, j].reshape(-1, 1)
            kij_sum = np.sum(rbf_kernel(vi, vj, gamma=gamma))
            dij = (kM[i] + kM[j] - 2 * kij_sum) ** power
            dist[i, j] = dist[j, i] = dij

    # 6) Final Agglomerative Clustering
    clustering = AgglomerativeClustering(n_clusters=n_Clust, metric="precomputed", linkage="average")
    labels = clustering.fit_predict(dist)

    return {
        "labels": labels,
        "persistence": persistence,
        "dist_matrix": dist,
    }
