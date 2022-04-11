import numpy as np
# import scipy.special 

def cheb(n):
    """
    Returns the (n+1)*(n+1) differentiation matrix D and (n+1) Chebyshev nodes
    x for the standard 1D interval [-1, 1]. The matrix multiplies a vector of
    function values at these nodes to give an approximation to the vector of
    derivative values. Nodes are output in descending order from 1 to -1. 

    Parameters:

    Returns:
    """
    
    if n == 0:
        x = 1
        D = 0
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

def osc_step(x0, h, y0, dy0, n, o):
    """
    Advances the solution from x0 to x0+h, starting from the initial conditions
    y(x0) = y0, y'(x0) = dy0. It uses the Barnett series of order o (up to and
    including the (o)th correction), and the underlying functions w, g are
    represented on an n-node Chebyshev grid.

    """
    pass

# May need n as arg
def nonosc_step(x0, h, y0, dy0, o):
    """

    """
    pass

def solve(w, g, xi, xf, yi, dyi, eps = 1e-6, xeval = []):
    """
    Solves y'' + 2gy' + w^2y = 0 on the interval (xi, xf), starting from the
    initial conditions y(xi) = yi, y'(xi) = dyi. Keeps the residual of the ODE
    below eps, and returns an interpolated solution (dense output) at points
    xeval.

    Parameters:

    Returns:
    """

    pass

def choose_n():
    """

    """
    pass

#def choose_o():
#    """
#
#    """
#    pass
