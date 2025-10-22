# %%
# Electrical Potential in Plate Capacitor (FEM)
# Translation of MATLAB Live Script to Python (Jupyter)
# Author: ChatGPT (GPT-5)
# Date: 2025-10-21

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu
from scipy.linalg import lu
import matplotlib.tri as tri

# LU decomposition
def lu_solve(A, b):
    """
    Solve Ax = b using LU decomposition via SciPy.
    Returns the solution x.
    """
    # Compute LU decomposition with pivoting
    P, L, U = lu(A)  # P @ A = L @ U

    # Solve Ly = P @ b (forward substitution)
    y = np.linalg.solve(L, P @ b)

    # Solve Ux = y (backward substitution)
    x = np.linalg.solve(U, y)

    return x

# Cholesky decomposition
def cholesky_solve(A, b):
    """Solve Ax = b using Cholesky decomposition (for SPD matrices)."""
    L = np.linalg.cholesky(A)
    y = np.linalg.solve(L, b)
    return np.linalg.solve(L.T, y)

# %%
# ============================================================
# === Gaussian Quadrature (Integration Points)
# ============================================================

def gauss_legendre(n):
    if n == 3:
        IP = np.array([
            [1/6, 1/6],
            [2/3, 1/6],
            [1/6, 2/3]
        ])
        w = np.array([1/3, 1/3, 1/3])
    elif n == 7:
        IP = np.array([
            [0.1012865073235, 0.1012865073235],
            [0.7974269853531, 0.1012865073235],
            [0.1012865073235, 0.7974269853535],
            [0.4701420641051, 0.0597158717898],
            [0.4701420641051, 0.4701420641051],
            [0.0597158717898, 0.4701420641051],
            [0.3333333333333, 0.3333333333333]
        ])
        w = np.array([
            0.1259391805448,
            0.1259391805448,
            0.1259391805448,
            0.1323941527885,
            0.1323941527885,
            0.1323941527885,
            0.225
        ])
    else:
        raise ValueError("Unsupported number of integration points.")
    return IP, w

# %%
# ============================================================
# === Element Routine: LST (Quadratic Triangular Element)
# ============================================================

def LST(x, u, matProp):
    """Compute element stiffness matrix and internal force for a 6-node triangle."""
    k = np.zeros((6, 6))
    fin = np.zeros(6)
    IP, w = gauss_legendre(3)

    for i in range(len(w)):
        xi, eta = IP[i]

        dNdxi = np.array([
            [-1+4*xi, 0, -3+4*(xi+eta), 4*eta, -4*eta, 4*(1-2*xi-eta)],
            [0, -1+4*eta, -3+4*(xi+eta), 4*xi, 4*(1-2*eta-xi), -4*xi]
        ])
        J = dNdxi @ x
        Jinv = np.linalg.inv(J)
        dNdx = Jinv @ dNdxi
        B = np.vstack([dNdx[0, :], dNdx[1, :]])
        detJ = np.linalg.det(J)

        N = np.array([
            -xi+2*xi**2,
            -eta+2*eta**2,
            1-3*xi-3*eta+2*xi**2+4*xi*eta+2*eta**2,
            4*xi*eta,
            4*(eta-eta**2-eta*xi),
            4*(xi-xi**2-xi*eta)
        ])
        uIP = N @ u
        lam = matProp(uIP)
        dlam = (matProp(uIP + 1e-8) - lam) / 1e-8

        fin += lam * (B.T @ (B @ u)) * w[i] * detJ
        k += (dlam * np.outer(B.T @ (B @ u), N) + lam * (B.T @ B)) * w[i] * detJ

    return fin, k

# %%
# ============================================================
# === FEM Assembly
# ============================================================

def FEM2D(FE, u2):
    """Assemble global stiffness matrix and force vector."""
    numN = FE['N'].shape[0]
    indP = FE['BC'][:, 0].astype(int) - 1  # MATLAB -> Python
    indF = np.setdiff1d(np.arange(numN), indP)

    u = np.zeros(numN)
    u[indF] = u2
    u[indP] = FE['BC'][:, 1]

    fin = np.zeros(numN)
    K = np.zeros((numN, numN))

    for e in FE['E']:
        xe = FE['N'][e, :]
        ue = u[e]
        fe, ke = LST(xe, ue, FE['matProp'])
        fin[e] += fe
        K[np.ix_(e, e)] += ke

    J = K[np.ix_(indF, indF)]
    f = -fin[indF]
    return f, J

# %%
# ============================================================
# === Visualization Functions
# ============================================================

def preProc(FE):
    """Plot the mesh discretization."""
    E = FE['E']
    N = FE['N']
    plt.figure(figsize=(6,6))
    for e in E:
        ind = [e[0], e[3], e[1], e[4], e[2], e[5], e[0]]
        coords = N[ind, :]
        plt.fill(coords[:,0], coords[:,1], color=[0, 0.4471, 0.7412], alpha=0.4, edgecolor='k', linewidth=0.5)
    plt.axis('equal')
    plt.axis('off')
    plt.title("Mesh Discretization", fontsize=12)
    plt.show()

def postProc(FE, u, title_str="Electrical Potential"):
    """
    Plot the electrical potential with color map and isolines.
    """
    N = FE['N']
    E = FE['E']

    # Build triangulation (Matplotlib expects 0-based indices)
    # For quadratic triangles (6 nodes), use only the 3 corner nodes [0,1,2]
    triang = tri.Triangulation(N[:, 0], N[:, 1], E[:, [0, 1, 2]])

    plt.figure(figsize=(6, 6))

    # Filled contour (smooth potential field)
    tpc = plt.tripcolor(triang, u, cmap='jet', shading='gouraud')
    plt.colorbar(tpc, label='Potential [V]')

    # Plot isolines (equipotential lines)
    levels = np.linspace(np.min(u), np.max(u), 15)  # number of contour lines
    cs = plt.tricontour(triang, u, levels=levels, colors='k', linewidths=0.6, alpha=0.8)
    plt.clabel(cs, inline=True, fontsize=8, fmt="%.2f", colors='black')

    # Mesh edges
    plt.triplot(triang, color='gray', linewidth=0.4, alpha=0.5)

    # Formatting
    plt.axis('equal')
    plt.axis('off')
    plt.title(title_str)
    plt.tight_layout()
    plt.show()

# %%
# ============================================================
# === Main Script
# ============================================================

# Load data (MATLAB struct)
FE_mat = scipy.io.loadmat('capacitor_small.mat', squeeze_me=True, struct_as_record=False)
FE_struct = FE_mat['FE']
FE = {
    'N': FE_struct.N,
    'E': FE_struct.E.astype(int) - 1,   # Fix: MATLAB → Python indices
    'BC': FE_struct.BC.astype(float),
    'matProp': lambda x: np.interp(x, [0, 1], [1, 1])
}

# Plot the mesh
preProc(FE)

# **************************************************************
# Assemble FEM matrices
b, A = FEM2D(FE, np.zeros(FE['N'].shape[0] - FE['BC'].shape[0]))

# *** SOLVE LINEAR SYSTEM ***
x = lu_solve(A, b)

# **************************************************************

# Postprocessing
m = FE['N'].shape[0]
u = np.zeros(m)
ind_free = np.setdiff1d(np.arange(m), FE['BC'][:, 0].astype(int) - 1)
u[ind_free] = x
u[(FE['BC'][:, 0].astype(int) - 1)] = FE['BC'][:, 1]
# Plot electrical potential
postProc(FE, u, "Electrical Potential")

print("✅ FEM solution completed successfully.")
