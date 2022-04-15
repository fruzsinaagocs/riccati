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

#def choose_n_cheb(x0, h, eps_wg, n0 = 16, nmax = 128):
#    """
#    Chooses the number of Chebyshev nodes to represent the functions w(x), g(x)
#    with over the interval (x0, x0+h). Doubles the number of points on every
#    iteration starting from n0 until the maximum relative error in w, g
#    (eps_wg) or nmax is reached. 
#    """
#    pass

# TODO: make o adaptive
def osc_step(w, x0, h, y0, dy0, o = 4, n = 16):
    """
    Advances the solution from x0 to x0+h, starting from the initial conditions
    y(x0) = y0, y'(x0) = dy0. It uses the Barnett series of order o (up to and
    including the (o)th correction), and the underlying functions w, g are
    represented on an n-node Chebyshev grid.

    """
    D, x = cheb(n)
    ws = w(h/2*x + x0 + h/2)
    w2 = ws**2
    y = 1j*ws
    R = lambda y: 2/h*np.matmul(D, y) + y**2 + w2
    Ry = 0
    maxerr = 0
    for i in range(1, o+1):
        y = y - Ry/(2*y)
        Ry = R(y)       
        maxerr = max(np.abs(Ry))
        #print("At iteration {}, max residual is Rx={}".format(i, maxerr))
    du1 = y
    du2 = np.conj(y)
    u1 = h/2*np.linalg.solve(D, du1)
    u1 -= u1[-1]
    u2 = np.conj(u1)
    f1 = np.exp(u1)
    f2 = np.conj(f1)
    C = np.array([[1, 1],[du1[-1], du2[-1]]])
    ap, am = np.linalg.solve(C, np.array([y0, dy0]))
    y1 = ap*f1 + am*f2
    dy1 = ap*du1*f1 + am*du2*f2
    return y1[0], dy1[0], maxerr 


def nonosc_step(x0, h, y0, dy0, o):
    """
    May need n as arg
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

#def choose_o():
#    """
#
#    """
#    pass
