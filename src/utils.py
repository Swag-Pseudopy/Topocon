import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

def plot_tsne_grid(X, results_dict, dataset_name):
    """Generates a grid of t-SNE plots for all 10 methods."""
    methods = list(results_dict.keys())
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle(f"Clustering Comparison: {dataset_name}", fontsize=16)
    
    # Precompute t-SNE for consistency across plots
    tsne = TSNE(n_components=2, random_state=42)
    X_embedded = tsne.fit_transform(X)
    
    for i, ax in enumerate(axes.flat):
        if i < len(methods):
            m_name = methods[i]
            labels = results_dict[m_name]
            ax.scatter(X_embedded[:, 0], X_embedded[:, 1], c=labels, cmap='tab10', s=5)
            ax.set_title(m_name)
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig(f"results/tsne_{dataset_name}.png")
    plt.close()
