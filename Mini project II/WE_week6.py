import numpy as np
import scipy.optimize as opt
import scipy.integrate as integrate
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt

# --- Exercise 6.4: Direct Energy Minimization for Geodesics ---
def energy_length(curve, metric):
    """Computes the energy functional for a given curve parameterized as an array of points."""
    total_energy = 0
    for i in range(len(curve) - 1):
        diff = curve[i+1] - curve[i]
        mid_point = (curve[i] + curve[i+1]) / 2
        total_energy += np.dot(diff.T, metric(mid_point) @ diff)
    return total_energy

def optimize_geodesic(straight_curve, metric):
    """Minimizes the energy functional to obtain the geodesic."""
    result = opt.minimize(lambda curve: energy_length(curve.reshape(-1, 2), metric), straight_curve.flatten(), method='L-BFGS-B')
    return result.x.reshape(-1, 2)

# Example quadratic metric
quad_metric = lambda x: (1 + np.linalg.norm(x)**2) * np.eye(2)

# Define endpoints
x1 = np.array([1, 1])
x2 = np.array([2, 3])

# Initial straight-line path
t_values = np.linspace(0, 1, 10)
initial_curve = np.outer(1 - t_values, x1) + np.outer(t_values, x2)

# Optimize
optimized_curve = optimize_geodesic(initial_curve, quad_metric)

# Plot result
plt.plot(initial_curve[:, 0], initial_curve[:, 1], 'r--', label='Initial Straight Line')
plt.plot(optimized_curve[:, 0], optimized_curve[:, 1], 'b-', label='Optimized Geodesic')
plt.scatter([x1[0], x2[0]], [x1[1], x2[1]], color='black', label='Endpoints')
plt.legend()
plt.show()

# --- Exercise 6.5: Density-Based Metric Geodesics ---
def density_metric(x, points, sigma=0.1, epsilon=1e-4):
    """Computes density metric at x based on dataset."""
    p_x = np.mean([np.exp(-euclidean(x, p)**2 / (2 * sigma**2)) for p in points])
    return np.eye(2) / (p_x + epsilon)

# Load dataset (toybanana.npy should be downloaded separately)
# toybanana = np.load('toybanana.npy')

# Compute geodesic under density metric
# optimized_curve_density = optimize_geodesic(initial_curve, lambda x: density_metric(x, toybanana))

# Extend this for third-order polynomial parametrization (omitted for brevity)
