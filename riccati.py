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

def choose_stepsize(w, x0, h, epsh = 1e-14, p = 32):
    """
    Chooses the stepsize h over which the functions w(x), g(x) can be
    represented sufficiently accurately. p/2 nodes are randomly chosen over the
    interval [x0, x0+h] where p is the number of Chebyshev nodes osc_step()
    uses, i.e. the number of nodes that will be used to compute the Barnett
    series. The interpolated w, g are then checked at these points and compared
    to the actual function values. If the largest relative error in w, g
    exceeds epsh, h is halved.
    TODO: Actually add g, so far only have w
    """
    q = p//2
    t = np.random.uniform(low = x0, high = x0+h, size = q)
    #print("Random nodes are: {}".format(t))
    s = x0 + h/2 + h/2*cheb(p-1)[1]
    V = np.ones((p, p))
    R = np.ones((q, p))
    for j in range(1, p):
        V[:, j] = V[:, j-1]*s
        R[:, j] = R[:, j-1]*t
    L = np.linalg.solve(V.T, R.T).T
    #print("L: ", L)
    #print("R: ", R)
    #print("R.T: ", R.T)
    wana = w(t)
    west = np.matmul(L, w(s))
    #print("Analytic w: ", wana)
    #print("Estimated w: ", west)
    maxwerr = max(np.abs((west - wana)/west))
    #print("Maximum interp error: ", maxwerr)
    if maxwerr > epsh:
        print("Stepsize h = {} is too large with max error {}".format(h, maxwerr))
        return choose_stepsize(w, x0, 0.7*h, epsh = epsh, p = p)
    else:
        print("Chose stepsize h = {}".format(h))
        return h
    #TODO: what if h is too small to begin with?

def osc_step(w, x0, h, y0, dy0, epsres = 1e-12, n = 32):
    """
    Advances the solution from x0 to x0+h, starting from the initial conditions
    y(x0) = y0, y'(x0) = dy0. It uses the Barnett series of order o (up to and
    including the (o)th correction), and the underlying functions w, g are
    represented on an n-node Chebyshev grid.

    """
    success = True
    ddy0 = -w(x0)**2*y0
    D, x = cheb(n)
    ws = w(h/2*x + x0 + h/2)
    w2 = ws**2
    y = 1j*ws
    R = lambda y: 2/h*np.matmul(D, y) + y**2 + w2
    Ry = 0
    maxerr = 10*epsres
    prev_err = np.inf
    o = 0 # Keep track of number of terms
    while maxerr > epsres:
        o += 1
        y = y - Ry/(2*y)
        Ry = R(y)       
        maxerr = max(np.abs(Ry))
        if maxerr >= prev_err:
            print("Barnett series diverged after {} terms".format(o-1))
            success = False
            #TODO: Actually fail here
            break
        prev_err = maxerr
        #print("At iteration {}, max residual is Rx={}".format(o, maxerr))
    if success:
        print("Converged after {} terms".format(o))
    print("Residue = {}".format(maxerr))
    du1 = y
    du2 = np.conj(du1)
    ddu1 = 2/h*np.matmul(D, y)
    ddu2 = np.conj(ddu1)
    u1 = h/2*np.linalg.solve(D, du1)
    u1 -= u1[-1]
    u2 = np.conj(u1)
    f1 = np.exp(u1)
    f2 = np.conj(f1)
    ddf1 = ddu1 + du1**2
    ddf2 = np.conj(ddf1)
    ap = (dy0 - du2[-1]*y0)/(du1[-1] - du2[-1])
    am = (dy0 - du1[-1]*y0)/(du2[-1] - du1[-1])
    bp = (ddy0*du2[-1] - dy0*ddf2[-1])/(ddf1[-1]*du2[-1] - ddf2[-1]*du1[-1])
    bm = (ddy0*du1[-1] - dy0*ddf1[-1])/(ddf2[-1]*du1[-1] - ddf1[-1]*du2[-1])
    y1 = ap*f1 + am*f2
    dy1 = bp*du1*f1 + bm*du2*f2
#    dy1 = ap*du1*f1 + am*du2*f2
    return y1[0], dy1[0], maxerr, success


def nonosc_step(x0, h, y0, dy0, o):
    """
    May need n as arg
    """
    pass

def solve(w, g, xi, xf, yi, dyi, eps = 1e-12, xeval = []):
    """
    Solves y'' + 2gy' + w^2y = 0 on the interval (xi, xf), starting from the
    initial conditions y(xi) = yi, y'(xi) = dyi. Keeps the residual of the ODE
    below eps, and returns an interpolated solution (dense output) at points
    xeval.

    Parameters:

    Returns:
    """
    # TODO: backwards integration
    epsh = 1e-13
    xs = [xi]
    ys = [yi]
    dys = [dyi]
    successes = [True]
    y = yi
    dy = dyi
    n = 32
    D, x = cheb(n)
    wi = w(xi)
    dwi = 2*np.matmul(D, w(xi + 1/2 + 1/2*x))[-1] 
    hi = wi/dwi
    print("Initial step: ", hi)
    h = choose_stepsize(w, xi, hi, epsh = epsh)
    xcurrent = xi
    while xcurrent < xf:
        print("x = {}, h = {}".format(xcurrent, h))
        # Check if we are at the end of solution range
        #if xcurrent + h > xf:
        #    h = xf - xcurrent
        # Attempt osc step of size h (for now always successful)
        y, dy, res, success = osc_step(w, xcurrent, h, y, dy, epsres = eps)
        # Log step
        ys.append(y)
        dys.append(dy)
        xs.append(xcurrent + h)
        successes.append(success)
        # Advance independent variable and choose next step
        wnext = w(xcurrent + h)
        dwnext = 2/h*np.matmul(D, w(xcurrent + h/2 + h/2*x))[0]
        hnext = wnext/dwnext
        xcurrent += h
        h = choose_stepsize(w, xcurrent, hnext, epsh = epsh)
        # TODO: update stepsize
    return xs, ys, dys, successes








