import numpy as np
import trace
import inspect

# *** Gauss elimination ***
def GaussEL(A, b):
    """
    Solves A x = b using Gaussian Elimination.

    """
    # size of the system
    n = len(b)

    L = np.eye(n)

    # Forward elimination
    for i in range(n - 1):
        for j in range(i + 1, n):
            lji = A[j, i] / A[i, i]

            for k in range(i, n):
                A[j, k] -= lji * A[i, k]

            b[j] -lji * b[i]

            L[j,i] = lji

    # Back substitution
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        s = 0.

        for j in range(i + 1, n):
            s += A[i, j] * x[j]

        x[i] = (b[i].item() - s) / A[i, i]

    return x

# *** LU decomposition ***
def LU(A, b):
    """
    Solve A x = b 
    
    Parameters
    ----------
    A : (n, n) numpy array
        Coefficient matrix (modified in-place).
    b : (n,) numpy array
        Right-hand side vector.
        
    Returns
    -------
    x : (n,) numpy array
        Solution vector such that A @ x = b.
    """
    A = A.copy().astype(float)
    b = b.astype(float)
    n = A.shape[0]

    # --- FACTORIZATION ---
    for i in range(n - 1):
        for j in range(i + 1, n):
            lji = A[j, i] / A[i, i]
            for k in range(i + 1, n):
                A[j, k] = A[j, k] - A[i, k] * lji
            A[j, i] = lji  # store L in lower part of A

    # --- FORWARD SUBSTITUTION ---
    y = np.zeros(n)
    for i in range(n):
        c = 0.0
        for j in range(i):
            c += A[i, j] * y[j]
        y[i] = b[i] - c

    # --- BACKWARD SUBSTITUTION ---
    x = np.zeros(n)
    x[-1] = y[-1] / A[-1, -1]
    for i in range(n - 2, -1, -1):
        c = 0.0
        for j in range(i + 1, n):
            c += A[i, j] * x[j]
        x[i] = (y[i] - c) / A[i, i]

    return x

# *** Cholesky decompostion ***
def CHOL(A, b):
    """
    Solve A x = b using Cholesky factorization (A = L * Lᵀ)
    
    Parameters
    ----------
    A : (n, n) numpy array
        Symmetric positive definite matrix (modified in-place).
    b : (n,) numpy array
        Right-hand side vector.
        
    Returns
    -------
    x : (n,) numpy array
        Solution vector.
    C : (n, n) numpy array
        Lower triangular Cholesky factor (same as L).
    """
    A = A.copy().astype(float)
    b = b.astype(float)
    n = A.shape[0]
    C = np.zeros_like(A)

    # --- FACTORIZATION ---
    for i in range(n):
        # diagonal entry
        c = 0.0
        for k in range(i):
            c += A[i, k] ** 2
        A[i, i] = np.sqrt(A[i, i] - c)

        # elements below diagonal
        for j in range(i + 1, n):
            c = 0.0
            for k in range(i):
                c += A[i, k] * A[j, k]
            A[j, i] = (A[j, i] - c) / A[i, i]

            # store for output
            C[j, i] = A[j, i]

    # --- FORWARD SUBSTITUTION: L y = b ---
    y = np.zeros(n)
    for i in range(n):
        c = 0.0
        for j in range(i):
            c += A[i, j] * y[j]
        y[i] = (b[i] - c) / A[i, i]

    # --- BACKWARD SUBSTITUTION: Lᵀ x = y ---
    x = np.zeros(n)
    x[-1] = y[-1] / A[-1, -1]
    for i in range(n - 2, -1, -1):
        c = 0.0
        for j in range(i + 1, n):
            c += A[j, i] * x[j]
        x[i] = (y[i] - c) / A[i, i]

    return x

# *** generate spd matrix ***
def generate_spd_matrix(n, seed=None):
    """
    Generate a random symmetric positive definite (SPD) matrix of size n x n.
    
    Parameters
    ----------
    n : int
        Dimension of the matrix.
    seed : int, optional
        Random seed for reproducibility.
        
    Returns
    -------
    A : (n, n) numpy array
        Symmetric positive definite matrix.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Random matrix
    M = np.random.randn(n, n)
    
    # Construct SPD matrix: A = M^T * M + n*I ensures positive definiteness
    A = np.dot(M.T, M) + n * np.eye(n)
    
    return A

def performance_solver(solver_func):
    """Run the given solver on a random system for performance tracing."""
    n = 100
    if solver_func == 'CHOL':
        A = generate_spd_matrix(n)
    else:
        A = np.random.rand(n, n)
    b = np.random.rand(n)
    solver_func(A, b)

# count lines for performance
def count_Lines(method='GaussEL'):
    """
    Count executed lines in the specified solver function (GaussEL, LU, or CHOL).

    Parameters
    ----------
    method : str
        One of 'GaussEL', 'LU', or 'CHOL'.
    """
    # --- Select solver function ---
    if method == 'GaussEL':
        solver_func = GaussEL
    elif method == 'LU':
        solver_func = LU
    elif method == 'CHOL':
        solver_func = CHOL
    else:
        raise ValueError("method must be one of: 'GaussEL', 'LU', or 'CHOL'")

    # --- Create tracer and run function ---
    tracer = trace.Trace(count=True, trace=False)
    tracer.runfunc(performance_solver, solver_func)  # ✅ FIXED: use runfunc()

    results = tracer.results()

    # --- Get source lines ---
    filename = inspect.getsourcefile(solver_func)
    lineno_start = solver_func.__code__.co_firstlineno
    source_lines, _ = inspect.getsourcelines(solver_func)

    print(f"\nLine execution counts for function {method} (file: {filename}):\n")
    for i, line in enumerate(source_lines, start=lineno_start):
        count = results.counts.get((filename, i), 0)
        print(f"{i:3d}: {count:5d} | {line.rstrip()}")


# *** TEST LINEAR SOLVER ***
# generate random arrays
n = 5
A = np.random.rand(n, n)
b = np.random.rand(n)

# seclect solver
solver = 'GaussEL'

# solve by Gauss
if solver == 'GaussEL':
    x = GaussEL(A, b)
    err = np.linalg.norm(A @ x - b)
    print(f"error in the solution (GAUSS): {err:5.4e}")
elif solver == 'LU':
    x = LU(A, b)
    err = np.linalg.norm(A @ x - b)
    print(f"error in the solution (LU): {err:5.4e}")
elif solver == 'CHOL':
    x = CHOL(A, b)
    print(x)
    err = np.linalg.norm(A @ x - b)
    print(f"error in the solution (CHOL): {err:5.4e}")

# count lines
#count_Lines(method = solver)


