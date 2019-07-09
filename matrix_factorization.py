"""
Matrix factorization functions
"""

import numpy as np
import cvxpy as cvx
import scipy.sparse as sparse
from sklearn.decomposition import SparsePCA
from tqdm import trange


def compressed_factorization(data, rank, fn, proj, nonneg=False, **kwargs):
    """Compute matrix factorization of compressed data PX.
    
    Args:
        data: (d, m) np.ndarray, compressed data matrix.
        rank: int, rank of factorization.
        fn: Callable, factorization function taking matrix and rank (np.ndarray, int) as first two arguments.
        proj: (d, n) np.ndarray or sparse matrix, projection/measurement matrix applied to data.
        nonneg: bool, if true, then set negative entries of recovered factor to zero.
        kwargs: Additional named arguments to be passed to factorization and recovery functions.
        
    Returns:
        Dictionary containing factors W ((n, r) np.ndarray), PW ((d, r) np.ndarray) and  H ((r, m) np.ndarray)
    """
    PW, H = fn(data, rank, **kwargs)
    assert(PW.shape[1] == H.shape[0])
    W = recover_lp(PW, proj, **kwargs)
    if nonneg:
        W[W < 0] = 0.
    return {'W': W, 'PW': PW, 'H': H}


def recover_lp(PW, P, solver=cvx.SCS, show_progress=False, verbose=False, **kwargs):
    """Sparse recovery via linear programming. Solves an equality-constrained L1-minimization problem
    for each column of the input matrix PW.
    
    Args:
        PW: (d, r) np.ndarray, compressed matrix.
        P: (d, n) np.ndarray or sparse matrix, projection/measurement matrix applied to data.
        solver: cvxpy solver.
        show_progress: bool, if true, show a progress bar over sparse recovery instances.
        verbose: bool, verbosity parameter for cvxpy solver.
    
    Returns:
        Recovered matrix, a (n, r) np.ndarray
    """
    d, r = PW.shape
    _, n = P.shape
    assert(P.shape[0] == d)
    
    _range = trange if show_progress else range
    
    if verbose:
        print(f'Solving {r} sparse recovery instances of dimension {n}')
    
    W = np.zeros([n, r], dtype=PW.dtype)
    for i in _range(r):
        # project to range of P
        y = P.dot(sparse.linalg.lsqr(P, PW[:, i])[0])
        
        # solve sparse recovery problem
        x = cvx.Variable(n)
        prob = cvx.Problem(
            cvx.Minimize(cvx.norm(x, 1)), 
            [P * x == y])
        prob.solve(solver=solver)
        W[:, i] = x.value.flat
        
    return W


def _get_random_state(random_state):
    if type(random_state) == np.random.RandomState:
        random = random_state
    else:
        random = np.random.RandomState(random_state)
    return random


# NMF PGD implementation based on https://www.csie.ntu.edu.tw/~cjlin/nmf/others/nmf.py

def nmf(X, rank, Winit=None, Hinit=None, iters=1000, l1_W=0., l1_H=0., show_progress=False, verbose=False, 
        random_state=None, **kwargs):
    """Compute an NMF of matrix X."""
    # initialize if necessary
    n, m = X.shape
    init_scale = np.sqrt(X.mean() / rank)
    random = _get_random_state(random_state)
    if Winit is None:
        Winit = init_scale * np.abs(random.randn(n, rank).astype(X.dtype))
    if Hinit is None:
        Hinit = init_scale * np.abs(random.randn(rank, m).astype(X.dtype))
        
    assert(Winit.shape == (n, rank))
    assert(Hinit.shape == (rank, m))
        
    W, H = Winit, Hinit
    alphaW, alphaH = 1., 1.
    
    _range = trange if show_progress else range
    
    if verbose:
        print(f'Computing rank {rank} NMF of matrix of shape ({n}, {m})')
    
    for i in _range(iters):
        W, alphaW, iterW = nlssubprob(X.T, H.T, W.T, alphaW, l1_reg=l1_W)
        W = W.T
        H, alphaH, iterH = nlssubprob(X, W, H, alphaH, l1_reg=l1_H)
        
    return W, H


def nlssubprob(X, W, Hinit, alpha, l1_reg=0., max_ls_iters=10): 
    """Compute a non-negative least squares projected gradient update on the right factor."""
    H = Hinit
    beta = 0.1
    WtX = np.dot(W.T, X)
    WtW = np.dot(W.T, W) 
    grad = np.dot(WtW, H) - WtX + l1_reg
    
    # backtracking line search for step size
    for i in range(max_ls_iters):
        Hn = H - alpha * grad
        Hn = np.where(Hn > 0, Hn, 0)
        d = Hn - H
        gradd = np.dot(grad.flat, d.flat)
        dQd = np.dot(np.dot(WtW, d).flat, d.flat)
        
        # check for sufficient improvement in objective
        suff_decr = (0.99 * gradd + 0.5 * dQd) < 0
        if i == 0:
            decr_alpha = not suff_decr
            Hp = H

        if decr_alpha: 
            if suff_decr:
                H = Hn; break
            else:
                alpha *= beta;
        else:
            if not suff_decr or (Hp == Hn).all():
                H = Hp; break
            else: 
                alpha /= beta
                Hp = Hn
                
    return (H, alpha, i+1)


# Sparse PCA

def sparse_pca(X, rank, alpha=1., ridge_alpha=1e-3, tol=1e-4, random_state=None, verbose=False, **kwargs):
    """Compute a Sparse PCA factorization of matrix X."""
    random = _get_random_state(random_state)
    n, m = X.shape
    if verbose:
        print(f'Computing rank {rank} sparse PCA of matrix of shape ({n}, {m}) with L1 parameter {alpha}')
    spca = SparsePCA(n_components=rank, alpha=alpha, ridge_alpha=ridge_alpha, tol=tol, random_state=random_state)
    spca.fit(X.T)
    W = spca.components_.T
    H = spca.transform(X.T).T
    return W, H
