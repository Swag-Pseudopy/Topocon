import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, MeanShift, SpectralClustering
from gudhi.clustering.tomato import Tomato
from src.topocon import TopoCon
from src.rcc import RCC
from src.bcc import BCC
from src.topoKmeans import topo_kmeans
from src.data_gen import get_mobius_torus, get_cylinder_torus
from src.utils import plot_tsne_grid

def run_all_methods(X, n_clusters):
    """Executes all 10 algorithms on the provided data."""
    results = {}
    
    # 1-4: Sklearn Baselines
    results['KMeans'] = KMeans(n_clusters=n_clusters).fit_predict(X)
    results['DBSCAN'] = DBSCAN(eps=0.5).fit_predict(X)
    results['MeanShift'] = MeanShift().fit_predict(X)
    results['Spectral'] = SpectralClustering(n_clusters=n_clusters).fit_predict(X)
    
    # 5: Tomato (Paper Ref: [cite: 311])
    # Note: Requires density estimation
    tomato = Tomato(n_clusters=n_clusters)
    results['Tomato'] = tomato.fit_predict(X)
    
    # 6: TopoKMeans (From your repo [cite: 44])
    results['TopoKMeans'] = topo_kmeans(X, nKNN=10, nClust=n_clusters)['labels']
    
    # 7: Convex Clustering (Standard Baseline [cite: 302])
    # Reusing RCC logic with beta=0 (no outlier component)
    results['ConvexClustering'] = RCC(alpha=0.1, beta=0).fit_predict(X, n_clusters)
    
    # 8: BiConvex Clustering (Paper: bcc.pdf [cite: 807, 1031])
    results['BiConvex'] = BCC(gamma=10.0, lam=0.1).fit_predict(X, n_clusters)
    
    # 9: Robust Convex Clustering (Paper: rcc.pdf [cite: 437, 531])
    results['RCC'] = RCC(alpha=0.1, beta=0.5).fit_predict(X, n_clusters)
    
    # 10: TopoCon (Paper: topocc.pdf [cite: 1, 165])
    results['TopoCon'] = TopoCon(n_clusters=n_clusters).fit_predict(X)
    
    return results

# EXECUTION BLOCK
if __name__ == "__main__":
    # Task: Synthetic (Mobius)
    data_mob, labels_mob = get_mobius_torus()
    res_mob = run_all_methods(data_mob, n_clusters=2)
    plot_tsne_grid(data_mob, res_mob, "Mobius_Torus")

    # Task: Real (Zoo)
    if os.path.exists('data/zoo.csv'):
        zoo_df = pd.read_csv('data/zoo.csv')
        X_zoo = zoo_df.drop(['animal_name', 'class_type'], axis=1).values
        y_zoo = zoo_df['class_type'].values
        res_zoo = run_all_methods(X_zoo, n_clusters=len(np.unique(y_zoo)))
        plot_tsne_grid(X_zoo, res_zoo, "Zoo_Dataset")
