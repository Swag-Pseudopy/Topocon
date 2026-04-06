import numpy as np

def get_mobius_torus(n_points=2000):
    """Generates a Mobius Torus as described in TopoCon paper."""
    u = np.random.uniform(0, 2*np.pi, n_points)
    v = np.random.uniform(0, 2*np.pi, n_points)
    # Mobius transformation logic
    x = (5 + (1 + 0.5 * np.cos(u/2)) * np.cos(v)) * np.cos(u)
    y = (5 + (1 + 0.5 * np.cos(u/2)) * np.cos(v)) * np.sin(u)
    z = 0.5 * np.sin(u/2) * np.cos(v)
    data = np.vstack([x, y, z]).T
    # Labels based on quadrants for synthetic ground truth
    labels = (u > np.pi).astype(int)
    return data, labels

def get_cylinder_torus(n_points=2000):
    """Generates intertwined Cylinder and Torus."""
    n_cyl = n_points // 2
    n_tor = n_points - n_cyl
    
    # Cylinder
    theta_c = np.random.uniform(0, 2*np.pi, n_cyl)
    z_c = np.random.uniform(-5, 5, n_cyl)
    x_c, y_c = 2 * np.cos(theta_c), 2 * np.sin(theta_c)
    
    # Torus
    u, v = np.random.uniform(0, 2*np.pi, n_tor), np.random.uniform(0, 2*np.pi, n_tor)
    x_t = (5 + 2 * np.cos(v)) * np.cos(u)
    y_t = (5 + 2 * np.cos(v)) * np.sin(u)
    z_t = 2 * np.sin(v)
    
    data = np.vstack([np.column_stack([x_c, y_c, z_c]), np.column_stack([x_t, y_t, z_t])])
    labels = np.array([0]*n_cyl + [1]*n_tor)
    return data, labels
