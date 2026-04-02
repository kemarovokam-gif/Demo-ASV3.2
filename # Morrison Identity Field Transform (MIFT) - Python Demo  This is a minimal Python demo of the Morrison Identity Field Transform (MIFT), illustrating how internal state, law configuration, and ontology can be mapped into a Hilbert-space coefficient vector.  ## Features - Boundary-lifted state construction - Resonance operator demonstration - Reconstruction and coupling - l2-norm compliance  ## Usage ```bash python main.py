import numpy as np

# ----------------------------
# Example Hilbert Space Setup
# ----------------------------

# X: internal state space (e.g., R^N)
N = 100
X = np.zeros(N)  # initial state vector

# L: law configuration (for demo, simple vector)
L = np.ones(N) * 0.1

# Omega: ontology configuration (for demo, simple vector)
Omega = np.linspace(0, 1, N)

# ----------------------------
# Operators
# ----------------------------

# ϕ_i: bounded linear functionals (for demo, pick standard basis functionals)
def phi_i(x, i):
    # returns the i-th component of x as linear functional
    return x[i]

# T_omega0: resonance operator (example: simple diagonal scaling)
def T_omega0(x, omega):
    return x * (1 + 0.5 * np.sin(2 * np.pi * omega))

# R: reconstruction operator (example: small perturbation)
def R(b):
    return 0.1 * b  # scaled boundary input

# b: boundary map (maps ontology × law -> boundary vector)
def b(omega, L):
    return np.sin(2*np.pi*omega) + 0.5*L

# coupling parameter
alpha = 0.2

# ----------------------------
# MIFT Transform
# ----------------------------

def xbnd(x, omega, L, alpha=alpha):
    """Boundary-lifted state"""
    return x + alpha * R(b(omega, L))

def MIFT(x, L, omega):
    """Morrison Identity Field Transform"""
    x_b = xbnd(x, omega, L)
    coeffs = np.array([phi_i(T_omega0(x_b, omega), i) for i in range(len(x))])
    # Ensure it belongs to l2
    assert np.isfinite(np.linalg.norm(coeffs)), "Coefficient sequence not in l2"
    return coeffs

# ----------------------------
# Example Usage
# ----------------------------

x0 = np.random.rand(N)  # random initial state
coeff_vector = MIFT(x0, L, Omega)

print("MIFT coefficient vector (first 10 entries):")
print(coeff_vector[:10])
print("l2 norm of coefficient vector:", np.linalg.norm(coeff_vector))
