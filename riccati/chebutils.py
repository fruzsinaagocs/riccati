import numpy as np

def coeffs2vals(coeffs):
    """
    Convert the Chebyshev coefficient representation of a set of polynomials `P_j` to their 
    values at Chebyshev nodes of the second kind (ordered from +1 to -1). This function returns a
    matrix `V` such that for an input coefficient matrix `C`,

    .. math:: V_{ij} = P_j(x_i) = \sum_{k=0}^{n} C_{kj}T_k(x_i).

    Taken from the `coeff2vals`_ function in the chebfun package.

    .. _`coeff2vals`: https://github.com/chebfun/chebfun/blob/master/%40chebtech2/coeffs2vals.m

    Parameters
    ----------
    coeffs: numpy.ndarray [float (real)]
       An array of size (n+1, m), with the (i, j)th element representing the
       projection of the jth input polynomial onto the ith Chebyshev
       polynomial.


    Returns
    -------
    values: np.ndarray [float (real)]
        An array of size (n+1, m), with the (i, j)th element representing the
        jth input polynomial evaluated at the ith Chebyshev node.

    """
    n = coeffs.shape[0]
    if n <= 1:
        values = coeffs
    else:
        coeffs[1:n-1,:] /= 2.0
        tmp = np.vstack((coeffs, coeffs[n-2:0:-1,:])) 
        values = np.real(np.fft.fft(tmp, axis = 0))
        values = values[:n,:]
    return values

def vals2coeffs(values):
    """
    Convert a matrix of values of `m` polynomials evaluated at `n+1` Chebyshev
    nodes of the second kind, ordered from +1 to -1, to their interpolating
    Chebyshev coefficients. This function returns the coefficient matrix `C`
    such that for an input matrix of values `V`,

    .. math:: F_j(x) = \sum_{k=0}^{n} C_{kj}T_k(x) 

    interpolates the values :math:`[V_{0j}, V_{1j}, \ldots, V_{nj}]` for :math:`j = 0..(m-1)`.

    Taken from the `vals2coeffs`_ function in the chebfun package.

    .. _`vals2coeffs`: https://github.com/chebfun/chebfun/blob/master/%40chebtech2/vals2coeffs.m

    Parameters
    ----------
    values: np.ndarray [float (real)]
        An array of size (n+1, m), with the (i, j)th element being the jth
        polynomial evaluated at the ith Chebyshev node. 

    Returns
    -------
    coeffs: np.ndarray [float (real)] 
        An array of size (n+1, m) with the (i, j)th element being the
        coefficient multiplying the ith Chebyshev polynomial for interpolating
        the jth input polynomial.

    """
    n = values.shape[0]
    if n <= 1:
        coeffs = values
    else:
        tmp = np.vstack((values[:n-1], values[n-1:0:-1]))
        coeffs = np.real(np.fft.ifft(tmp, axis = 0))
        coeffs = coeffs[0:n,:]
        coeffs[1:n-1,:] *= 2
    return coeffs

def integrationm(n):
    """
    Chebyshev integration matrix. It maps function values at n Chebyshev nodes
    of the second kind, ordered from +1 to -1, to values of the integral of the
    interpolating polynomial at those points, with the last value (start of the
    interval) being zero. Taken from the `cumsummat`_ function in the chebfun package.

    .. _`cumsummat`: https://github.com/chebfun/chebfun/blob/master/%40chebcolloc2/chebcolloc2.m

    Parameters
    ----------
    n: int
        Number of Chebyshev nodes the integrand is evaluated at. Note the nodes
        are always ordered from +1 to -1.

    Returns
    -------
    Q: np.ndarray [float (real)]
        Integration matrix of size (n, n) that maps values of the integrand at
        the n Chebyshev nodes to values of the definite integral on the
        interval, up to each of the Chebyshev nodes (the last value being zero by definition). 

    """
    n -= 1
    T = coeffs2vals(np.identity(n+1))
    Tinv = vals2coeffs(np.identity(n+1))
    k = np.linspace(1.0, n, n)
    k2 = 2*(k-1)
    k2[0] = 1
    B = np.diag(1/(2*k), -1) - np.diag(1/k2, 1)
    v = np.ones(n)
    v[1::2] = -1
    B[0,:] = sum(np.diag(v) @ B[1:n+1,:], 0)
    B[:,0] *= 2
    Q = T @ B @ Tinv
    Q[-1,:] = 0
    return Q

def quadwts(n):
    """
    Clenshaw-Curtis quadrature weights mapping function evaluations at
    (n+1) Chebyshev nodes of the second kind, ordered from +1 to -1, to value of the
    definite integral of the interpolating function on the same interval. Taken
    from [1]_ Ch 12, `clencurt.m`

    Parameters
    ----------
    n: int
        The (number of Chebyshev nodes - 1) for which the quadrature weights are to be computed. 

    Returns
    -------
    w: np.ndarray [float (real)]
        Array of size (n+1,) containing the quadrature weights.

    References
    ----------
    .. [1] Trefethen, Lloyd N. Spectral methods in MATLAB. Society for
      industrial and applied mathematics, 2000.

    """
    if n == 0:
        w = 0
    else:
        a = np.linspace(0.0, np.pi, n+1)
        w = np.zeros(n+1)
        v = np.ones(n-1)
        if n % 2 == 0:
            w[0] = 1.0/(n**2 - 1)
            w[n] = w[0]
            for k in range(1, n//2):
                v = v - 2*np.cos(2*k*a[1:-1])/(4*k**2 - 1)
            v -= np.cos(n*a[1:-1])/(n**2 - 1)
        else:
            w[0] = 1.0/n**2
            w[n] = w[0]
            for k in range(1,(n+1)//2):
                v -= 2*np.cos(2*k*a[1:-1])/(4*k**2 - 1)
        w[1:-1] = 2*v/n
    return w


def cheb(n):
    """
    Returns a differentiation matrix D of size (n+1, n+1) and (n+1) Chebyshev
    nodes x for the standard 1D interval [-1, 1]. The matrix multiplies a
    vector of function values at these nodes to give an approximation to the
    vector of derivative values. Nodes are output in descending order from 1 to
    -1. The nodes are given by
    
    .. math:: x_p = \cos \left( \\frac{\pi n}{p} \\right), \quad n = 0, 1, \ldots p.

    Parameters
    ----------
    n: int
        Number of Chebyshev nodes - 1.

    Returns
    -------
    D: numpy.ndarray [float]
        Array of size (n+1, n+1) specifying the differentiation matrix.
    x: numpy.ndarray [float]
        Array of size (n+1,) containing the Chebyshev nodes.
    """
    if n == 0:
        x = 1
        D = 0
        w = 0
    else:
        a = np.linspace(0.0, np.pi, n+1)
        x = np.cos(a)
        b = np.ones_like(x)
        b[0] = 2
        b[-1] = 2
        d = np.ones_like(b)
        d[1::2] = -1
        c = b*d
        X = np.outer(x, np.ones(n+1))
        dX = X - X.T
        D = np.outer(c, 1/c) / (dX + np.identity(n+1))
        D = D - np.diag(D.sum(axis=1))
    return D, x

def interp(s, t):
    """
    Creates interpolation matrix from an array of source nodes s to target nodes t.
    Taken from `here`_ .

    .. _`here`: https://github.com/ahbarnett/BIE3D/blob/master/utils/interpmat_1d.m


    Parameters
    ----------
    s: numpy.ndarray [float]
        Array specifying the source nodes, at which the function values are known.
    t: numpy.ndarray [float]
        Array specifying the target nodes, at which the function values are to
        be interpolated.

    Returns
    -------
    L: numpy.ndarray [float]
        Array defining the inteprolation matrix L, which takes function values
        at the source points s and yields the function evaluated at target
        points t. If s has size (p,) and t has size (q,), then L has size (q, p).
    """
    r = s.shape[0]
    q = t.shape[0]
    V = np.ones((r, r))
    R = np.ones((q, r))
    for j in range(1, r):
        V[:, j] = V[:, j-1]*s
        R[:, j] = R[:, j-1]*t
    L = np.linalg.solve(V.T, R.T).T
    return L

def spectral_cheb(info, x0, h, y0, dy0, niter):
    """
    Utility function to apply a spectral collocation method based on Chebyshev nodes from
    x = x0 to x = x0+h, starting from the initial conditions y(x0) = y0, y'(x0)
    = dy0. In each spectral collocation/Chebyshev step, the solver iterates
    over how many nodes to perform the step with, starting from `info.nini`,
    doubling in every iteration until `info.nmax` is reached or the required
    tolerance is satisfied, whichever comes sooner. The `niter` parameter keeps
    track of how many iterations have been done and is used to retrieve the
    relevant pre-computed differentiation matrix and vector of nodes from the
    input `info` object.

    Parameters
    ----------
    info: Solverinfo object

    x0: float (real)
        Value of the independent variable to take a spectral step from.

    h: float (real)
        Stepsize.

    y0, dy0: complex
        Initial conditions (value of the solution and its derivative) at x0.

    niter: int
        Integer to keep track of how many iterations of the spectral collocation step have been done.

    Returns
    -------
    y1, dy1: complex
        Numerical estimate of the solution and its derivative at the end of the step, at x0+h.

    xscaled: numpy.ndarray [float]
        Chebyshev nodes used for the current iteration of the spectral
        collocation method scaled to lie between [x0, x0+h].

    """
    D, x = info.Ds[niter], info.nodes[niter]
    xscaled = h/2*x + x0 + h/2
    ws = info.w(xscaled)
    gs = info.g(xscaled)
    w2 = ws**2
    D2 = (D @ D) + h*(np.diag(gs) @ D) + h**2/4*np.diag(w2)
    n = round(info.ns[niter])
    ic = np.zeros(n+1, dtype=complex)
    ic[-1] = 1 # Because nodes are ordered backwards, [1, -1]
    D2ic = np.zeros((n+3, n+1), dtype=complex)
    D2ic[:n+1] = D2
    D2ic[-2] = D[-1] 
    D2ic[-1] = ic
    rhs = np.zeros(n+3, dtype=complex)
    rhs[-2] = dy0*h/2
    rhs[-1] = y0
    y1, res, rank, sing = np.linalg.lstsq(D2ic, rhs) # NumPy solve only works for square matrices
    dy1 = 2/h*(D @ y1)
    info.increase(LS = 1)
    return y1, dy1, xscaled


