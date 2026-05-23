# Topological Convex Clustering (TopoCon)

This repository contains the official Python implementation of **TopoCon: Where Persistent Homology meets Convex Optimization**.

TopoCon is a novel clustering framework that seamlessly integrates persistent homology from Topological Data Analysis (TDA) with convex optimization. By embedding multiscale topological features (derived from Vietoris-Rips filtrations) into a topology-aware convex clustering model, TopoCon accurately captures intrinsic manifold structures and non-convex geometries that traditional distance-based algorithms miss.

## 📖 Table of Contents

* [Overview](https://www.google.com/search?q=%23overview)
* [Key Contributions](https://www.google.com/search?q=%23key-contributions)
* [Repository Structure](https://www.google.com/search?q=%23repository-structure)
* [Installation](https://www.google.com/search?q=%23installation)
* [Usage](https://www.google.com/search?q=%23usage)
* [Experimental Results](https://www.google.com/search?q=%23experimental-results)
* [Ablation Studies](https://www.google.com/search?q=%23ablation-studies)
* [Citation](https://www.google.com/search?q=%23citation)

## 🔬 Overview

Traditional clustering methods (like k-means, Spectral Clustering) often fail to capture complex structures like nested, multiscale, or intertwined configurations. While standard Convex Clustering guarantees global optimality, it operates purely geometrically and ignores higher-order data structures like loops or voids.

TopoCon bridges this gap by:

1. Constructing local neighborhoods via the $k$-nearest neighbor ($k$-NN) algorithm.


2. Extracting topological features (connected components $H_0$ and loops $H_1$) using Vietoris-Rips filtrations.


3. Embedding these features into a unified Persistence Matrix.


4. Solving a strongly convex optimization problem using an ADMM-based solver, weighted by a Gaussian kernel similarity matrix to preserve local topological consistency.



**The TopoCon Objective Function:**


$$ \min_{U \in \mathbb{R}^{M \times n}} \left[ \frac{1}{2}\|P - U\|_F^2 + \nu \sum_{i<j} w_{ij}\|U_i - U_j\|_2 + \frac{\gamma}{2}\|U\|_F^2 \right] $$


*(Where $P$ is the persistence matrix, $w_{ij}$ are Gaussian kernel weights, $\nu$ controls fusion strength, and $\gamma$ is the shrinkage parameter)*.

## ✨ Key Contributions

* **Global Convergence:** The optimization problem is strictly convex, guaranteeing a unique global minimum free from initialization biases.


* **Topology-Aware Fusion:** Centroids merge smoothly based on multiscale topological similarity rather than just Euclidean proximity.


* **Robustness:** Inherits the Lipschitz stability of persistence diagrams, making it highly resilient to structural Gaussian noise.



## 📂 Repository Structure

The codebase is organized as follows[cite: 7]:

```text
├── LICENSE
├── README.md
├── main_pipeline.py      # Main entry point to execute experiments
├── run_all.sh            # Shell script to automate the full pipeline execution
└── src/
    ├── topocon.py        # Core implementation of Topological Convex Clustering
    ├── bcc.py            # Baseline: Bi-Convex Clustering (BCC)
    ├── rcc.py            # Baseline: Robust Convex Clustering (RCC)
    ├── topokmeans.py     # Baseline: Topological k-Means (TKM)
    ├── data_gen.py       # Generators for complex synthetic manifolds
    └── utils.py          # Helper functions for metrics, kernels, and graphing

```

## ⚙️ Installation

To run the pipeline, ensure you have Python installed along with the required dependencies. We recommend setting up a virtual environment.

```bash
git clone https://github.com/yourusername/Topocon.git
cd Topocon
pip install numpy scipy scikit-learn cvxpy ripser matplotlib

```

## 🚀 Usage

You can run individual experiments or the entire benchmark suite using the provided pipeline scripts.

**Run the full experimental pipeline:**

```bash
bash run_all.sh

```

**Run a specific module via Python:**

```bash
python main_pipeline.py --dataset cylinder_torus --method topocon

```

## 📊 Experimental Results

TopoCon was evaluated against a rigorous suite of baseline methods including KM, DBSCAN, Mean-Shift, Spectral Clustering, CC, BCC, RCC, TKM, ToMATo, and TPCC.

### Synthetic Manifolds

On highly complex, intertwined topological structures, TopoCon achieves perfect or near-perfect Adjusted Rand Index (ARI) scores:

* **Cylinder Torus:** ARI = 1.000


* **Möbius Torus:** ARI = 1.000


* **Two Moons:** ARI = 0.951


* **Torus Line Sphere:** ARI = 0.942



*(Place your 3D visual panels here)*

> `![Qualitative Synthetic Results](assets/Images/panel_cylinder_torus.png)`
> `![Mobius Torus Comparison](assets/Images/clustering_comparison_panel.png)`

### Real-World Datasets

TopoCon generalizes efficiently to high-dimensional real-world data across tabular, biological, and image domains:

* **Wisconsin Breast Cancer:** ARI = 0.821


* **ORHD:** ARI = 0.783


* **Zoo:** ARI = 0.772



*(Place your t-SNE visualizations here)*

> `![t-SNE Real Data](assets/Images/tsne_wisconsin.png)`

## 🔬 Ablation Studies

Extensive ablation studies confirm the necessity of the core components:

1. **Homological Dimensions:** For synthetic manifolds with intrinsic loops, combining both $H_0$ and $H_1$ persistence pairs is critical. Dropping either dimension results in significant performance decay.


2. **Topology-Aware Similarity:** Replacing the Gaussian kernel with a uniform weight ($w_{ij} = 1$) drastically reduces clustering quality, proving the necessity of topology-guided fusion paths.


3. **Noise Robustness:** TopoCon retains strong topological separation capabilities even under substantial Gaussian noise perturbations ($\rho = 0.3$), consistently peaking at moderate local neighborhood graph sizes ($k$).



*(Place your ablation and robustness curves here)*

> `![Hyperparameter Ablation](assets/Images/parameter_ablation_3d.png)`
> `![Noise Robustness](assets/Images/noise_robustness_ablation.png)`

## 📜 Citation

If you find this code useful in your research, please consider citing our paper:

```bibtex
@article{pratihar2026topocon,
  title={Topological Convex Clustering: Where Persistent Homology meets Convex Optimization},
  author={Pratihar, Arghya and Das, Swagato and Das, Swagatam},
  journal={Indian Statistical Institute, Kolkata},
  year={2026}
}

```
