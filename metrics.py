"""
Metrics for evaluating the quality of a factorization.
"""

import numpy as np
import cvxpy as cvx


def error(X, Y, normalize=True, norm='fro'):
    """Error ||X - Y||. If `normalize` is true, the error is divided by ||X||."""
    err = np.linalg.norm(X - Y, norm)
    if normalize:
        err /= np.linalg.norm(X, norm)
    return err


def mean_absolute_correlation(X, Y, match_rows=False):
    if not match_rows:
        X, Y = X.T, Y.T
    assert(X.shape[0] == Y.shape[0])
    corr = np.mean([absolute_correlation(X[i], Y[i]) for i in range(X.shape[0])])
    return corr


def absolute_correlation(x, y):
    """Absolute value of the Pearson correlation between vectors x and y."""
    x0 = x - x.mean()
    y0 = y - y.mean()
    r = np.abs(x0.dot(y0))
    return r / np.linalg.norm(x0) / np.linalg.norm(y0)


def l2_similarity(x, y):
    """The negative sign-invariant squared L2 distance between vectors x and y."""
    return -min(np.linalg.norm(x - y)**2., np.linalg.norm(x + y)**2.)


def _match_factors(X, Y, similarity, match_rows=False):
    """Compute a maximum bipartite matching between the rows of X and rows of Y with
    the given similarity function. If match_rows is False, matches columns instead.
    
    Returns:
        Dict containing match and result. match[i] is the row (column, if match_rows is True) 
        of Y matched to row i of X, and result is the similarity score corresponding 
        to this matching.
    """
    if not match_rows:
        X, Y = X.T, Y.T
    
    assert(X.shape[0] == Y.shape[0])
    k = X.shape[0]
    
    # construct optimization problem
    sims = np.array([[similarity(X[i], Y[j]) for j in range(k)] for i in range(k)])
    x = cvx.Variable(k, k)
    z = 0
    for i in range(k):
        for j in range(k):
            z += x[i, j] * sims[i, j]
    obj = cvx.Maximize(z)
    constraints = []
    for i in range(k):
        constraints.append(cvx.sum_entries(x[i, :]) == 1)
        constraints.append(cvx.sum_entries(x[:, i]) == 1)
    constraints.append(x >= 0)
    
    # solve problem and extract solution
    prob = cvx.Problem(obj, constraints)
    result = prob.solve()
    match = np.zeros(k, dtype=int)
    for i in range(k):
        match[i] = np.argmax(x.value[i])
    return {'match': match, 'score': result}


def match_correlation(X, Y, match_rows=False):
    """Return the maximum mean Pearson correlation between the columns (or rows) of X and Y
    over permutations of their columns (or rows)"""
    res = _match_factors(X, Y, absolute_correlation, match_rows)
    k = X.shape[0] if match_rows else X.shape[1]
    res['score'] = res['score'] / k
    return res


def match_l2(X, Y, match_rows=False, normalize=True):
    """Return the minimum Frobenius distance between X and Y over permutations of columns (or rows)."""
    res = _match_factors(X, Y, l2_similarity, match_rows)
    res['score'] = np.sqrt(-res['score'])
    if normalize:
        res['score'] = res['score'] / np.linalg.norm(X, 'fro')
    return res


def normalize_factors(W, H):
    """Normalize the columns of W and rows of H such that their product is the same,
    but the rows of H have unit norm."""
    assert(W.shape[1] == H.shape[0])
    W1 = np.zeros_like(W)
    H1 = np.zeros_like(H)
    for i in range(W.shape[1]):
        Hnorm = np.linalg.norm(H[i])
        H1[i] = H[i] / Hnorm
        W1[:, i] = W[:, i] * Hnorm
    return W1, H1


def permute_factors(X, match, permute_rows=True):
    """Permute the rows (columns if permute_rows is False) of X according to indices in match."""
    X_matched = np.zeros_like(X)
    for i in range(X.shape[0] if permute_rows else X.shape[1]):
        if permute_rows:
            X_matched[match[i]] = X[i]
        else:
            X_matched[:, match[i]] = X[:, i]
    return X_matched

