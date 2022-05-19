import numpy as np
import math
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

def interp(s, t):
    """
    Creates interpolation matrix from an array of source nodes s to target nodes t.
    Taken from https://github.com/ahbarnett/BIE3D/blob/master/utils/interpmat_1d.m
    """
    p = s.shape[0]
    q = t.shape[0]
    V = np.ones((p, p))
    R = np.ones((q, p))
    for j in range(1, p):
        V[:, j] = V[:, j-1]*s
        R[:, j] = R[:, j-1]*t
    L = np.linalg.solve(V.T, R.T).T
    return L

def choose_stepsize(w, g, x0, h, epsh = 1e-14, p = 32):
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
    a = np.linspace(np.pi/(2*p), np.pi*(1 - 1/(2*p)), p)
    t = x0 + h/2 + h/2*np.cos(a)
    s = x0 + h/2 + h/2*cheb(p)[1]
    L = interp(s, t)
    wana = w(t)
    west = np.matmul(L, w(s))
    gana = g(t)
    gest = np.matmul(L, g(s))
    maxwerr = max(np.abs((west - wana)/west))
    maxgerr = max(np.abs((gest - gana)/gest))
    maxerr = max(maxwerr, maxgerr)
    if maxerr > epsh:
        #print("Stepsize h = {} is too large with max error {}".format(h, maxwerr))
        return choose_stepsize(w, g, x0, 0.7*h, epsh = epsh, p = p)
    else:
        #print("Chose stepsize h = {}".format(h))
        return h
    #TODO: what if h is too small to begin with?

def osc_step(w, g, x0, h, y0, dy0, epsres = 1e-12, n = 32):
    """
    Advances the solution from x0 to x0+h, starting from the initial conditions
    y(x0) = y0, y'(x0) = dy0. It uses the Barnett series of order o (up to and
    including the (o)th correction), and the underlying functions w, g are
    represented on an n-node Chebyshev grid.

    """
    success = True
    D, x = cheb(n)
    xscaled = h/2*x + x0 + h/2
    ws = w(xscaled)
    gs = g(xscaled)
    w2 = ws**2
    g2 = gs**2
    y = -gs + np.lib.scimath.sqrt(g2 - w2) # Need to use complex sqrt
    #y = 1j*ws
    R = lambda y: 2/h*np.matmul(D, y) + y**2 + w2 + 2*gs*y
    Ry = 0
    maxerr = 10*epsres
    prev_err = np.inf
    o = 0 # Keep track of number of terms
    while maxerr > epsres:
        o += 1
        y = y - Ry/(2*(y + gs))
        Ry = R(y)       
        #uy = h/2*np.linalg.solve(D, y)
        #fuy = np.exp(uy - uy[-1])
        #print("fuy: ", fuy)
        #scaledres = max(np.abs(y0*Ry*fuy))
        #maxerr = scaledres
        #maxerr2 = max(np.abs(Ry))
        maxerr = max(np.abs(Ry))
        if maxerr >= prev_err:
            print("Barnett series diverged after {} terms".format(o-1))
            success = False
            #TODO: Actually fail here
            break
        prev_err = maxerr
        print("At iteration {}, riccati y = {}, max residual is Rx={}".format(o, y[0], maxerr))
        #print("At iteration {}, riccati y = {}, max residual is Rx={}, max scaled residual is {}".format(o, y[0], maxerr2, scaledres))
    if success:
        print("Converged after {} terms".format(o))
    print("Residue = {}".format(maxerr))
    du1 = y
    du2 = np.conj(du1)
    u1 = h/2*np.linalg.solve(D, du1)
    u1 -= u1[-1]
    u2 = np.conj(u1)
    f1 = np.exp(u1)
    f2 = np.conj(f1)
    C = np.array([[1, 1], [du1[-1], du2[-1]]])
    ap, am = np.linalg.solve(C, np.array([y0, dy0]))
    y1 = ap*f1 + am*f2
    dy1 = ap*du1*f1 + am*du2*f2
    phase = np.imag(u1[0])
    #print("at x = {}, y = {}, u = {}, f = {}, y1 = {}".format(x0, y, u1, f1, y1))
    print("y1:{}, dy1:{}".format(y1[0], dy1[0]))
    return y1[0], dy1[0], maxerr, success, phase

def nonosc_step(w, g, x0, h, y0, dy0, epsres = 1e-12, n = 16):

    """
    Advances the solution from x0 to x0+h, starting from the initial conditions
    y(x0) = y0, y'(x0) = dy0. It uses a Chebyshev spectral method with enough
    nodes to achieve epsres relative accuracy, starting from an initial n
    nodes.
    """
    success = True #TODO: this doesn't do anything in Cheb, can always add more nodes
    res = 0 #TODO: does nothing
    maxerr = 10*epsres
    N = 4 # TODO
    Nmax = 512 # TODO
    #print("Nonoscillatory step from x0={} y0={}, dy0={}, h={}".format(x0, y0, dy0, h))
    yprev, dyprev, xprev = spectral_cheb(w, g, x0, h, y0, dy0, N)
    while maxerr > epsres:
        #print("Trying n={} points".format(2*N))
        N *= 2
        if N > Nmax:
            #print("Chebyshev step didn't converge with n <= {}".format(Nmax))
            success = False
            return 0, 0, maxerr, success, res
        y, dy, x = spectral_cheb(w, g, x0, h, y0, dy0, N) 
        #L = interp(x, xprev)
        #yest = np.matmul(L, y)
        maxerr = np.abs((yprev[0] - y[0]))
        #print("Maxerr: ", maxerr, "y[0]: ", y[0])
        if np.isnan(maxerr):
            maxerr = np.inf
        yprev = y
        dyprev = dy
        xprev = x
    #print("Converged at n={} points, success = {}, y[0], dy[0] = {}, {}".format(N, success, y[0], dy[0]))
    return y[0], dy[0], maxerr, success, res

def spectral_cheb(w, g, x0, h, y0, dy0, n):
    """
    Utility function to apply a spectral method based on n Chebyshev nodes from
    x = x0 to x = x0+h, starting from the initial conditions y(x0) = y0, y'(x0)
    = dy0.
    """
    D, x = cheb(n)
    xscaled = h/2*x + x0 + h/2
    #print("xscaled: ", xscaled)
    ws = w(xscaled)
    gs = g(xscaled)
    w2 = ws**2
    D2 = 4/h**2*np.matmul(D, D) + 4/h*np.matmul(np.diag(gs), D) + np.diag(w2)
    ic = np.zeros(n+1, dtype=complex)
    ic[-1] = 1 # Because nodes are ordered backwards, [1, -1]
    D2ic = np.zeros((n+3, n+1), dtype=complex)
    D2ic[:n+1] = D2
    D2ic[-2] = 2/h*D[-1] 
    D2ic[-1] = ic
    rhs = np.zeros(n+3, dtype=complex)
    rhs[-2] = dy0
    rhs[-1] = y0
    y1, res, rank, sing = np.linalg.lstsq(D2ic, rhs) # NumPy solve only works for square matrices
    dy1 = 2/h*np.matmul(D, y1)
    return y1, dy1, xscaled


def solve(w, g, xi, xf, yi, dyi, eps = 1e-12, epsh = 1e-13, xeval = []):
    """
    Solves y'' + 2gy' + w^2y = 0 on the interval (xi, xf), starting from the
    initial conditions y(xi) = yi, y'(xi) = dyi. Keeps the residual of the ODE
    below eps, and returns an interpolated solution (dense output) at points
    xeval.

    Parameters:

    Returns:
    """
    # TODO: backwards integration
    xs = [xi]
    ys = [yi]
    dys = [dyi]
    phases = []
    steptypes = [0]
    successes = [True]
    y = yi
    dy = dyi
    yprev = y
    dyprev = dy
    n = 16 # How many points we use during Cheby interp
    p = n # How many points we use to choose h
    D, x = cheb(n)
    wi = w(xi)
    gi = g(xi)
    dwi = 2*np.matmul(D, w(xi + 1/2 + 1/2*x))[-1] 
    dgi = 2*np.matmul(D, g(xi + 1/2 + 1/2*x))[-1] 
    # Choose initial stepsize
    hslo = min(1, 1e8, np.abs(1/wi))
    hosc_ini = min(1e8, np.abs(wi/dwi), np.abs(gi/dgi))
    hosc = choose_stepsize(w, g, xi, hosc_ini, epsh = epsh)  
    xcurrent = xi
    wnext = wi
    dwnext = dwi
    while xcurrent < xf:
        print("x = {}, hosc = {}, hslo = {}".format(xcurrent, hosc, hslo))
        # Check how oscillatory the solution is
        ty = np.abs(1/wnext)
        tw = np.abs(wnext/dwnext)
        tw_ty = tw/ty
        print("Timescale ratio: {}".format(tw_ty))
        success = False
        if tw_ty > 10 and hosc*wnext/(2*np.pi) > 1:
            print("Attempting oscillatory step")
            # Solution is oscillatory
            # Attempt osc step of size hosc
            y, dy, res, success, phase = osc_step(w, g, xcurrent, hosc, yprev, dyprev, epsres = eps, n = n)
            print("Success? {}".format(success))
            if success == 1:
                print("Successful oscillatory step")
            print("y, dy = {}, {}".format(y, dy))
            steptype = 1
        while success == False:
            print("Attempting nonoscillatory step with h = {}".format(hslo)) 
            # Solution is not oscillatory, or previous step failed
            # Attempt Cheby step of size hslo
            y, dy, err, success, res = nonosc_step(w, g, xcurrent, hslo, yprev, dyprev, epsres = eps, n = n)
            phase = 0
            steptype = 0
            # If step still unsuccessful, halve stepsize
            if success == False:
                hslo *= 0.5
            else:
                print("Successful nonoscillatory step")
            if hslo < 1e-16:
                raise RuntimeError("Solution didn't converge between h = 1e-16")
        # Log step
        if steptype == 1:
            h = hosc
        else:
            h = hslo
        ys.append(y)
        dys.append(dy)
        xs.append(xcurrent + h)
        phases.append(phase)
        steptypes.append(steptype)
        successes.append(success) # TODO: now always true
        # Advance independent variable and compute next stepsizes
        wnext = w(xcurrent + h)
        gnext = g(xcurrent + h)
        dwnext = 2/h*np.matmul(D, w(xcurrent + h/2 + h/2*x))[0]
        dgnext = 2/h*np.matmul(D, g(xcurrent + h/2 + h/2*x))[0]
        xcurrent += h
        hslo = min(1e8, np.abs(1/wnext))
        hosc_ini = min(1e8, np.abs(wnext/dwnext), np.abs(gnext/dgnext))
        hosc = choose_stepsize(w, g, xcurrent, hosc_ini, epsh = epsh)  
        yprev = y
        dyprev = dy
    print('Done')
    return xs, ys, dys, successes, phases, steptypes








