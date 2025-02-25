def group_lasso_solver(Y, D, epsilon, lam, max_iter=100, tol=1e-4):
    """
    Solves for the coefficients A for a group S of patches using a proximal
    gradient method for the group Lasso problem:
    
      minimize   (1/|S|) sum_k ||A(k,:)||_2 + (lambda/2)||Y - D A||_F^2
      subject to  ||Y - D A||_F^2 <= epsilon
    
    Here Y is of shape (m, |S|) and D is of shape (m, K). 
    This formulation is a penalized version that approximately enforces the constraint.
    
    Args:
        Y (np.ndarray): Data matrix for group S, shape (m, s)
        D (np.ndarray): Current dictionary, shape (m, K)
        epsilon (float): Tolerance on the reconstruction error.
        lam (float): Regularization parameter.
        max_iter (int): Maximum number of iterations.
        tol (float): Tolerance for convergence.
        
    Returns:
        A (np.ndarray): Coefficient matrix for group S, shape (K, s)
    """
    m, s = Y.shape
    _, K = D.shape
    A = np.zeros((K, s))
    # Set step size (could use Lipschitz constant estimation)
    L = np.linalg.norm(D, ord=2)**2  # Lipschitz constant for D^T D
    t = 1.0 / L
    
    for it in range(max_iter):
        # Gradient step: gradient of (lambda/2)||Y-D A||_F^2 w.r.t. A
        grad = - D.T @ (Y - D @ A) * lam
        A_next = A - t * grad
        
        # Proximal operator for group Lasso: for each row
        for k in range(K):
            norm_row = np.linalg.norm(A_next[k, :])
            thresh = t / (s)  # note: 1/|S| weight; adjust t if needed
            if norm_row > thresh:
                A_next[k, :] = (1 - thresh / norm_row) * A_next[k, :]
            else:
                A_next[k, :] = 0.
                
        # Check convergence
        if np.linalg.norm(A_next - A) < tol:
            break
        A = A_next.copy()
        
    # (Optional) if reconstruction error > epsilon, one might increase lam.
    return A

def update_dictionary(Y_list, A_list, D, max_iter=50, tol=1e-4):
    """
    Updates the dictionary D given a set of groups of patches Y and their coefficients A.
    
    Args:
        Y_list (list): List of np.ndarray; each Y is shape (m, s_i) for group i.
        A_list (list): List of np.ndarray; each A is shape (K, s_i) for group i.
        D (np.ndarray): Current dictionary, shape (m, K).
        max_iter (int): Maximum iterations for dictionary update.
        tol (float): Tolerance for convergence.
        
    Returns:
        D_new (np.ndarray): Updated dictionary with unit ℓ2 columns.
    """
    m, K = D.shape
    # Aggregate gradient over all groups:
    grad = np.zeros_like(D)
    total_error = 0.0
    total_count = 0
    for Y, A in zip(Y_list, A_list):
        err = Y - D @ A
        grad += - err @ A.T
        total_error += np.linalg.norm(err, 'fro')**2
        total_count += 1
    # A simple gradient descent step
    step = 1e-3  # step size, may need tuning
    D_new = D - step * grad
    # Project each column to unit norm
    for k in range(K):
        D_new[:, k] /= max(np.linalg.norm(D_new[:, k]), 1e-8)
    return D_new

def solve_equation8(groups, initial_D, epsilon_list, lam=0.1, n_outer=10):
    """
    Solve Equation (8) using alternating minimization.
    
    Args:
        groups (list): List of groups. Each element is a tuple (Y, indices) where
                       Y is the matrix of patches for that group (each column a patch,
                       shape (m, s_i)), and indices is an optional identifier.
        initial_D (np.ndarray): Initial dictionary, shape (m, K)
        epsilon_list (list): List of epsilon_i for each group.
        lam (float): Regularization parameter for group sparse coding.
        n_outer (int): Number of alternating iterations.
        
    Returns:
        D (np.ndarray): Learned dictionary.
        A_groups (list): List of coefficient matrices for each group.
    """
    D = initial_D.copy()
    A_groups = [None] * len(groups)
    
    for outer in range(n_outer):
        Y_list = []
        # Sparse coding step: For each group solve for A_i
        for i, (Y, _) in enumerate(groups):
            epsilon_i = epsilon_list[i]
            # Y has shape (m, s_i)
            A = group_lasso_solver(Y, D, epsilon_i, lam)
            A_groups[i] = A
            Y_list.append(Y)
        # Dictionary update step: update D using all groups
        D = update_dictionary(Y_list, A_groups, D)
        # (Optional) monitor overall error:
        total_err = 0.0
        for i, (Y, _) in enumerate(groups):
            err = np.linalg.norm(Y - D @ A_groups[i], 'fro')**2
            total_err += err
        print(f'Iteration {outer}, total reconstruction error: {total_err:.4f}')
    return D, A_groups