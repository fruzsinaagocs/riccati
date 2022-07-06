import numpy as np
import math
# import scipy.special 

class counter:
    """
    Wraps a function to keep track of how many times it's been called.
    """

    def __init__(self, f):
        self.f = f
        self.count = 0

    def __call__(self, *args, **kwargs):
        self.count += 1
        return self.f(*args, **kwargs)

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
    global n_chebnodes
    n_chebnodes += 1
    return D, x

def interp(s, t):
    """
    Creates interpolation matrix from an array of source nodes s to target nodes t.
    Taken from https://github.com/ahbarnett/BIE3D/blob/master/utils/interpmat_1d.m
    """
    r = s.shape[0]
    q = t.shape[0]
    V = np.ones((r, r))
    R = np.ones((q, r))
    for j in range(1, r):
        V[:, j] = V[:, j-1]*s
        R[:, j] = R[:, j-1]*t
    L = np.linalg.solve(V.T, R.T).T
#    global n_LS
#    n_LS += 1
    return L

def choose_nonosc_stepsize(w, g, x0, h, p = 16):
    """

    """
#    print("choosing nonosc stepsize")
    xscaled = x0 + h/2 + h/2*xp
    ws = w(xscaled)
    if max(ws) > 1.2/h:
#        print("Reducing stepsize for nonosc step")
        return choose_nonosc_stepsize(w, g, x0, h/2)
    else:
#        print("Chose stepsize h=",h)
        return h

def choose_osc_stepsize(w, g, x0, h, epsh = 1e-12, p = 32):
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
    #print("choosing osc stepsize")
#    a = np.linspace(np.pi/(2*p), np.pi*(1 - 1/(2*p)), p)
    t = x0 + h/2 + h/2*xpinterp
    s = x0 + h/2 + h/2*xp
#    L = interp(s, t)
#    L = interp(xp, np.cos(a))
#    print("s, t: ", s[:], t[:])
#    print("mins, maxes (s, t):", min(s), max(s), min(t), max(t))
#    s2, t2 = xp, np.cos(a)
#    Lp = interp(s2, t2)
#    print("s/2, t/2: ", (s/2)[:2], (t/2)[:2])
#    print("max(L - Lp): ", max((L - Lp).flatten()))
    wana = w(t)
    west = np.matmul(L, w(s))
    gana = g(t)
    gest = np.matmul(L, g(s))
    maxwerr = max(np.abs((west - wana)/wana))
    maxgerr = max(np.abs((gest - gana)/gana))
    maxerr = max(maxwerr, maxgerr)
#    print("maxwerr, maxwerr2: ", maxwerr, maxwerr2)
#    print("maxgerr, maxgerr2: ", maxgerr, maxgerr2)
    if maxerr > epsh:
#        print("Stepsize h = {} is too large (epsh = {}) with max error {}".format(h, epsh, maxerr))
#        return choose_osc_stepsize(w, g, x0, min(0.7*h, 0.9*h*(epsh/maxerr)**(1/(p-1))), epsh = epsh, p = p)
        return choose_osc_stepsize(w, g, x0, min(0.7*h, 0.9*h*(epsh/maxerr)**(1/(p-1))), epsh = epsh, p = p)
    else:
#        print("Chose stepsize h = {}".format(h))
        return h
    #TODO: what if h is too small to begin with?

def osc_step(w, g, x0, h, y0, dy0, epsres = 1e-12, n = 32, plotting=False, k=0):
    """
    Advances the solution from x0 to x0+h, starting from the initial conditions
    y(x0) = y0, y'(x0) = dy0. It uses the Barnett series of order o (up to and
    including the (o)th correction), and the underlying functions w, g are
    represented on an n-node Chebyshev grid.

    """
#    print("taking osc step")
    success = True
    xscaled = h/2*xn + x0 + h/2
#    print(xscaled.shape)
    ws = w(xscaled)
    gs = g(xscaled)
    #y = -gs + np.lib.scimath.sqrt(g2 - w2) # Need to use complex sqrt
    y = 1j*ws
    delta = lambda r, y: -r/(2*(y + gs))
    R = lambda d: 2/h*np.matmul(Dn, d) + d**2
    Ry = 1j*2*(1/h*np.matmul(Dn, ws) + gs*ws)
    #maxerr = 10*epsres
    maxerr = max(np.abs(Ry))
    prev_err = np.inf
    o = 0 # Keep track of number of terms
    if plotting == False:
        while maxerr > epsres:
            o += 1
            deltay = delta(Ry, y)
            y = y + deltay
            Ry = R(deltay)       
            maxerr = max(np.abs(Ry))
            if maxerr >= prev_err:
#                print("Barnett series diverged after {} terms".format(o-1))
                success = False
                #TODO: Actually fail here
                break
            prev_err = maxerr
#            print("At iteration {}, riccati y = {}, max residual is Rx={}".format(o, y[0], maxerr))
        if success:
#            print("Converged after {} terms".format(o))
            pass
#        print("Residue = {}".format(maxerr))
    else:
        while o < k:
            o += 1
            deltay = delta(Ry, y)
            y = y + deltay
            Ry = R(deltay)       
            maxerr = max(np.abs(Ry))
#            print("At iteration {}, riccati y = {}, max residual is Rx={}".format(o, y[0], maxerr))
    du1 = y
    du2 = np.conj(du1)
    u1 = h/2*np.linalg.solve(Dn, du1)
    u1 -= u1[-1]
    u2 = np.conj(u1)
    f1 = np.exp(u1)
    f2 = np.conj(f1)
    C = np.array([[1, 1], [du1[-1], du2[-1]]])
    ap, am = np.linalg.solve(C, np.array([y0, dy0]))
#    ap = (dy0 - y0*du2[-1])/(du1[-1] - du2[-1])   
#    am = (dy0 - y0*du1[-1])/(du2[-1] - du1[-1])
    y1 = ap*f1 + am*f2
    dy1 = ap*du1*f1 + am*du2*f2
    phase = np.imag(u1[0])
#    print("at x = {}, y = {}, u = {}, f = {}, y1 = {}".format(x0, y, u1, f1, y1))
#    print("C matrix: ", C)
#    print("y0, dy0: ", y0, dy0)
#    print("ap, am: ", ap, am)
#    print("ap1, am1: ", ap1, am1)
#    print("y1:{}, dy1:{}".format(y1[0], dy1[0]))
    global n_LS, n_LS2x2, n_riccstep
    n_LS += 1
    n_LS2x2 += 1
    n_riccstep += 1
    if plotting:
        xplot = np.linspace(x0, x0+h, 500)
        L = interp(xscaled, xplot)
        u1plot = np.matmul(L, u1)
        f1plot = np.exp(u1plot)
        f2plot = np.conj(f1plot)
        yplot = ap*f1plot + am*f2plot
        return xplot, yplot, xscaled, y1, maxerr
    else:
        return y1[0], dy1[0], maxerr, success, phase

def nonosc_step(w, g, x0, h, y0, dy0, epsres = 1e-12, nmax = 64, nini = 16):

    """
    Advances the solution from x0 to x0+h, starting from the initial conditions
    y(x0) = y0, y'(x0) = dy0. It uses a Chebyshev spectral method with enough
    nodes to achieve epsres relative accuracy, starting from an initial n
    nodes.
    """
    success = True #TODO: this doesn't do anything in Cheb, can always add more nodes
    res = 0 #TODO: does nothing
    maxerr = 10*epsres
    N = nini
    Nmax = nmax
    #print("Nonoscillatory step from x0={} y0={}, dy0={}, h={}".format(x0, y0, dy0, h))
    yprev, dyprev, xprev = spectral_cheb(w, g, x0, h, y0, dy0, 0)
    while maxerr > epsres:
#        print("Trying n={} points".format(2*N))
        N *= 2
        if N > Nmax:
#            print("Chebyshev step didn't converge with n <= {}".format(Nmax))
            success = False
            return 0, 0, maxerr, success, res
        y, dy, x = spectral_cheb(w, g, x0, h, y0, dy0, int(np.log2(N/nini))) 
        #L = interp(x, xprev)
        #yest = np.matmul(L, y)
        maxerr = np.abs((yprev[0] - y[0]))
#        print("Maxerr: ", maxerr, "y[0]: ", y[0])
        if np.isnan(maxerr):
            maxerr = np.inf
        yprev = y
        dyprev = dy
        xprev = x
#    print("Converged at n={} points, success = {}, y[0], dy[0] = {}, {}".format(N, success, y[0], dy[0]))
    global n_chebstep
    n_chebstep += 1
    return y[0], dy[0], maxerr, success, res

def spectral_cheb(w, g, x0, h, y0, dy0, niter):
    """
    Utility function to apply a spectral method based on n Chebyshev nodes from
    x = x0 to x = x0+h, starting from the initial conditions y(x0) = y0, y'(x0)
    = dy0.
    """
#    global Ds, xs
    D, x = Ds[niter], nodes[niter]
#    print("niter:", niter)
    xscaled = h/2*x + x0 + h/2
    #print("xscaled: ", xscaled)
    ws = w(xscaled)
    gs = g(xscaled)
    w2 = ws**2
#    print("x, xscaled: ", x, xscaled)
#    print("w2, gs: ", w2, gs)
    D2 = 4/h**2*np.matmul(D, D) + 4/h*np.matmul(np.diag(gs), D) + np.diag(w2)
    n = round(ns[niter])
#    print("n: ", n)
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
    global n_LS, n_chebits
    n_LS += 1
    n_chebits += 1
    return y1, dy1, xscaled


def solve(w, g, xi, xf, yi, dyi, eps = 1e-12, epsh = 1e-12, xeval = [], n = 16, p = 16):
    """
    Solves y'' + 2gy' + w^2y = 0 on the interval (xi, xf), starting from the
    initial conditions y(xi) = yi, y'(xi) = dyi. Keeps the residual of the ODE
    below eps, and returns an interpolated solution (dense output) at points
    xeval.

    Parameters:

    Returns:
    """
    # Parameters
    nini = 16 # n to start from in Cheby steps
    nmax = 64 # maximum n in Cheby steps
    n = n # How many points we use during Cheby interp in Riccati steps, 16 for Airy, 32 for burst
    p = p # How many points we use to choose h
    hi = 0.1 # Initial stepsize for calculating derivatives

    # Counters for steps
    global n_riccstep, n_chebstep, n_chebnodes, n_LS, n_LS2x2, n_chebits
    n_riccstep, n_chebstep, n_chebnodes, n_LS, n_LS2x2, n_chebits = 0, 0, 0, 0, 0, 0

    # Store D matrices for Cheby steps
    global Dn, xn, Ds, nodes, ns, xp, xpinterp, L
    Dlength = int(np.log2(nmax/nini)) + 1
    Ds, nodes = [], []
    ns = np.geomspace(nini, nmax, Dlength)#, dtype=int)
    for i in range(Dlength - 1):
        D, x = cheb(nini*2**i)
        Ds.append(D)
        nodes.append(x)
    if n in ns:
        i = np.where(ns == n)[0][0]
        Dn, xn = Ds[i], nodes[i]
    else:
        Dn, xn = cheb(n)
    if p in ns:
        i = np.where(ns == p)[0][0]
        xp = nodes[i]
    else:
        xp = cheb(p)[1]
    xpinterp = np.cos(np.linspace(np.pi/(2*p), np.pi*(1 - 1/(2*p)), p))
    L = interp(xp, xpinterp)

#    print("ns: ", ns)

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
#    print('yprev, dyprev:', yprev, dyprev)
    wi = w(xi)
    gi = g(xi)
    dwi = 2/hi*np.matmul(Dn, w(xi + hi/2 + hi/2*xn))[-1] 
    dgi = 2/hi*np.matmul(Dn, g(xi + hi/2 + hi/2*xn))[-1] 
    # Choose initial stepsize
    hslo_ini = min(1e8, np.abs(1/wi))
    hosc_ini = min(1e8, np.abs(wi/dwi), np.abs(gi/dgi))
#    print("g/dg: ", gi/dgi, "w/dw: ", wi/dwi, "dw: ", dwi)
#    print("Initial step guesses (slo, osc): ", hslo_ini, hosc_ini)
    hslo = choose_nonosc_stepsize(w, g, xi, hslo_ini)
    hosc = choose_osc_stepsize(w, g, xi, hosc_ini, epsh = epsh, p = p)  
    xcurrent = xi
    wnext = wi
    dwnext = dwi
    while xcurrent < xf:
#        print("x = {}, hosc = {}, hslo = {}".format(xcurrent, hosc, hslo))
        # Check how oscillatory the solution is
        #ty = np.abs(1/wnext)
        #tw = np.abs(wnext/dwnext)
        #tw_ty = tw/ty
        #print("Timescale ratio: {}".format(tw_ty))
        success = False
        if hosc > hslo*5 and hosc*wnext/(2*np.pi) > 1:
#            print("Attempting oscillatory step")
            # Solution is oscillatory
            # Attempt osc step of size hosc
            y, dy, res, success, phase = osc_step(w, g, xcurrent, hosc, yprev, dyprev, epsres = eps, n = n)
#            print("Success? {}".format(success))
            if success == 1:
#                print("Successful oscillatory step")
#            print("y, dy = {}, {}".format(y, dy))
                pass
            steptype = 1
        while success == False:
#            print("Attempting nonoscillatory step with h = {}".format(hslo)) 
            # Solution is not oscillatory, or previous step failed
            # Attempt Cheby step of size hslo
            y, dy, err, success, res = nonosc_step(w, g, xcurrent, hslo, yprev, dyprev, epsres = eps, nini = nini, nmax = nmax)
            phase = 0
            steptype = 0
            # If step still unsuccessful, halve stepsize
            if success == False:
                hslo *= 0.5
            else:
#                print("Successful nonoscillatory step")
                pass
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
#        print("estimating next stepsize")
        wnext = w(xcurrent + h)
        gnext = g(xcurrent + h)
        dwnext = 2/h*np.matmul(Dn, w(xcurrent + h/2 + h/2*xn))[0]
        dgnext = 2/h*np.matmul(Dn, g(xcurrent + h/2 + h/2*xn))[0]
        xcurrent += h
        hslo_ini = min(1e8, np.abs(1/wnext))
        hosc_ini = min(1e8, np.abs(wnext/dwnext), np.abs(gnext/dgnext))
        hosc = choose_osc_stepsize(w, g, xcurrent, hosc_ini, epsh = epsh, p = p)  
        hslo = choose_nonosc_stepsize(w, g, xcurrent, hslo_ini)
        yprev = y
        dyprev = dy
    #print('Done')
    statdict = {"cheb steps": (n_chebstep, sum(np.array(steptypes) == 0) - 1), 
                "cheb iterations": n_chebits, 
                "ricc steps": (n_riccstep, sum(np.array(steptypes) == 1)), 
                "linear solves": n_LS, 
                "linear solves 2x2": n_LS2x2, 
                "cheb nodes": n_chebnodes}
    return xs, ys, dys, successes, phases, steptypes, statdict








