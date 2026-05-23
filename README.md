# Topological Convex Clustering (TopoCon)

This repository contains the official Python implementation of **TopoCon: Where Persistent Homology meets Convex Optimization**.

TopoCon is a novel clustering framework that seamlessly integrates persistent homology from Topological Data Analysis (TDA) with convex optimization. By embedding multiscale topological features (derived from Vietoris-Rips filtrations) into a topology-aware convex clustering model, TopoCon accurately captures intrinsic manifold structures and non-convex geometries that traditional distance-based algorithms miss.

## 🔬 Overview

Traditional clustering methods (like k-means, Spectral Clustering) often fail to capture complex structures like nested, multiscale, or intertwined configurations. While standard Convex Clustering guarantees global optimality, it operates purely geometrically and ignores higher-order data structures like loops or voids.

TopoCon bridges this gap by:

1. Constructing local neighborhoods via the $k$-nearest neighbor ($k$-NN) algorithm.


2. Extracting topological features (connected components $H_0$ and loops $H_1$) using Vietoris-Rips filtrations.


3. Embedding these features into a unified Persistence Matrix.


4. Solving a strongly convex optimization problem using an ADMM-based solver, weighted by a Gaussian kernel similarity matrix to preserve local topological consistency.



**The TopoCon Pipeline (Warmup):**
Below is the conceptual framework of TopoCon, illustrating the progression from local neighborhood filtrations to the persistence-matrix-driven convex optimization.

> `![TopoCon Framework/Warmup Diagram](assets/Images/warmup_diagram.png)`

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

TopoCon was evaluated against a suite of baseline methods including KM, DBSCAN, Mean-Shift, Spectral Clustering, CC, BCC, RCC, TKM, ToMATo, and TPCC.

> `![Mobius Torus Comparison](assets/Images/clustering_comparison_panel.png)`

### Synthetic Manifolds

On highly complex, intertwined topological structures, TopoCon achieves perfect or near-perfect Adjusted Rand Index (ARI) scores:

* **Cylinder Torus:** ARI = 1.000

> `![Cylinder Torus Synthetic Results](assets/Images/panel_cylinder_torus.png)`

* **Sphere Ring:** ARI = 0.914
  
> `![Sphere Ring Synthetic Results](assets/Images/panel_sphere_ring.png)`

* **Two Moons:** ARI = 0.951

> `![Two Moons Synthetic Results](assets/Images/panel_two_moons.png)`


### Real-World Datasets

TopoCon generalizes efficiently to high-dimensional real-world data across tabular, biological, and image domains. For instance, on the **Zoo dataset**, TopoCon successfully maps higher-dimensional features into coherent clusters ($\text{ARI} = 0.772$), demonstrating adaptability beyond controlled synthetic manifolds.

* **Wisconsin Breast Cancer:** ARI = 0.821

> `![Wisconsin B.C. Dataset t-SNE](assets/Images/tsne_wisconsin.png)`

* **ORHD:** ARI = 0.783

> [ORHD Dataset t-SNE](assets/Images/tsne_orhd.png)

* **Zoo:** ARI = 0.772

> `[Zoo Dataset t-SNE](assets/Images/tsne_zoo.png)`


## 🔬 Ablation Studies & Robustness

Extensive ablation studies confirm the necessity of the core components and evaluate the framework's behavior under stress.

### 1. Homological Dimensions ($H_0$ vs $H_1$)

To quantify the impact of different topological signatures, we evaluated TopoCon using only connected components ($H_0$), only loops ($H_1$), and their combination. While $H_0$ carries the primary clustering signal for dense real-world data, complex intertwined synthetic manifolds require the synergistic combination of both $H_0$ and $H_1$ to achieve peak performance.

> `![Homology Ablation](assets/Images/homology_ablation_panel.png)`

### 2. Topology-Aware Similarity

Replacing the Gaussian kernel with a uniform weight ($w_{ij} = 1$) drastically reduces clustering quality, proving the necessity of topology-guided fusion paths.

> `![Hyperparameter Ablation](assets/Images/parameter_ablation_3d.png)`

### 3. Noise Robustness (Torus-Line-Sphere Extremes)

We tested TopoCon's resilience on the **Torus-Line-Sphere** dataset under varying levels of Gaussian noise ($\rho$). The performance curve demonstrates stable structural separation at low noise levels, peaking at moderate local neighborhood graph sizes ($k$). However, as the graph becomes over-smoothed at excessively high $k$ values, the performance collapses chaotically as the structural voids are bridged by noise.

> `![Noise Robustness Curve](assets/Images/noise_robustness_ablation.png)`

To visually highlight this structural breakdown, the following panel compares the ground truth against TopoCon's predictions at two environmental extremes: the clean baseline ($\rho = 0.0, \text{ARI} = 0.989$) versus a highly noisy state ($\rho = 0.3, \text{ARI} = 0.326$) where spatial bleeding obscures the manifold boundaries.

> `![Torus Line Sphere Extremes](assets/Images/tls_noise_ablat.png)`

<!--## 📜 Citation

If you find this code useful in your research, please consider citing our paper:

```bibtex
@article{pratihar2026topocon,
  title={Topological Convex Clustering: Where Persistent Homology meets Convex Optimization},
  author={Pratihar, Arghya and Das, Swagato and Das, Swagatam},
  journal={Indian Statistical Institute, Kolkata},
  year={2026}
```
-->
