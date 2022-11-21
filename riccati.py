import numpy as np
import math
import scipy.linalg

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
    r = s.shape[0]
    q = t.shape[0]
    V = np.ones((r, r))
    R = np.ones((q, r))
    for j in range(1, r):
        V[:, j] = V[:, j-1]*s
        R[:, j] = R[:, j-1]*t
    L = np.linalg.solve(V.T, R.T).T
    return L

def choose_nonosc_stepsize(info, x0, h, epsh = 0.2):
    """
    Chooses the stepsize for spectral Chebyshev steps, based on the variation
    of 1/w, the approximate timescale over which the solution changes. If over
    the suggested interval h 1/w changes by 20% or more, the interval is
    halved, otherwise it's accepted.
    """
    xscaled = x0 + h/2 + h/2*info.xp
    ws = info.w(xscaled)
    if max(ws) > (1 + epsh)/h:
        return choose_nonosc_stepsize(info, x0, h/2, epsh = epsh)
    else:
        return h

def choose_osc_stepsize(info, x0, h, epsh = 1e-12):
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
    w, g, L = info.w, info.g, info.L
    t = x0 + h/2 + h/2*info.xpinterp
    s = x0 + h/2 + h/2*info.xp
    if info.p == info.n:
        info.wn = w(s)
        info.gn = g(s)
        ws = info.wn
        gs = info.gn
    else:
        info.wn = w(x0 + h/2 + h/2*info.xn)
        info.gn = g(x0 + h/2 + h/2*info.xn)
        ws = w(s)
        gs = g(s)
    wana = w(t)
    west = L.dot(ws)
    gana = g(t)
    gest = L.dot(gs)
    maxwerr = max(np.abs((west - wana)/wana))
    maxgerr = max(np.abs((gest - gana)/gana))
    maxerr = max(maxwerr, maxgerr)
    if maxerr > epsh:
        return choose_osc_stepsize(info, x0, min(0.7*h, 0.9*h*(epsh/maxerr)**(1/(info.p-1))), epsh = epsh)
    else:
        return h
    #TODO: what if h is too small to begin with?


def osc_evolve(info, x0, x1, h, y0, epsres = 1e-12, epsh = 1e-12):
    """
    Allows continuous evolution between x0 and x1. Starting from x0 and y =
    [y0, yd0] takes a Riccati step of size h or until x1 is reached. 
    Will need to call choose_osc_stepsize() before the first call to this to
    set info.wn, info.gn correctly, or set them by hand.
    """
    # Make sure info's attributes are up-to-date:
    # success = 0
    # info.x = x0
    # info.y = y0
    # Check if we're stepping out of range
    sign = np.sign(h)
    if sign*(x0 + h) > sign*x1:
        # Step would be out of bounds, need to rescale and re-eval wn, gn
        h = x1 - x0
        xscaled = h/2*info.xn + x0 + h/2
        info.wn = info.w(xscaled)
        info.gn = info.g(xscaled)
    info.h = h
    # Call oscillatory step
    y10, y11, maxerr, s, phase = osc_step(info, x0, h, y0[0], y0[1], epsres = epsres)
    if s != 1:
        # Unsuccessful step
        success = 0
    else:
        # Successful step
        success = 1
        info.y = np.array([y10, y11])
        info.x += h
        # Determine new stepsize
        wnext = info.wn[0]
        gnext = info.gn[0]
        dwnext = 2/h*info.Dn.dot(info.wn)[0]
        dgnext = 2/h*info.Dn.dot(info.gn)[0]
        hosc_ini = min(1e8, np.abs(wnext/dwnext), np.abs(gnext/dgnext))
        info.h = choose_osc_stepsize(info, info.x, hosc_ini, epsh = epsh)  
    return success

def nonosc_evolve(info, x0, x1, h, y0, epsres = 1e-12, epsh = 0.2):
    """
    Allows continuous evolution between x0 and x1. Starting from x0 and y =
    [y0, yd0] takes a Chebyshev step of size h or until x1 is reached. 
    Will need to call choose_nonosc_stepsize() before the first call to this to
    set info.wn, info.gn correctly, or set them by hand.
    """

    # Make sure info's attributes are up-to-date:
    # success = 0
    # info.x = x0
    # info.y = y0
    # Check if we're stepping out of range
    sign = np.sign(h)
    if sign*(x0 + h) > sign*x1:
        # Step would be out of bounds, need to rescale and re-eval wn, gn
        h = x1 - x0
        xscaled = h/2*info.xn + x0 + h/2
    info.h = h
    # Call nonoscillatory step
    y10, y11, maxerr, s, res = nonosc_step(info, x0, h, y0[0], y0[1], epsres = epsres)
    if s != 1:
        # Unsuccessful step
        success = 0
    else:
        # Successful step
        success = 1
        info.y = np.array([y10, y11])
        info.x += h
        # Determine new stepsize
        wnext = info.w(info.x)
        hosc_ini = min(1e8, np.abs(1/wnext))
        info.h = choose_nonosc_stepsize(info, info.x, hosc_ini, epsh = epsh)  
    return success


def single_osc_step(info, x0, h, y0, dy0, epsres = 1e-12):
    """
    Standalone single Riccati step.
    Advances the solution from x0 to x0+h, starting from the initial conditions
    y(x0) = y0, y'(x0) = dy0. It uses the Barnett series of order o (up to and
    including the (o)th correction), and the underlying functions w, g are
    represented on an (n+1)-node Chebyshev grid.
    """
    success = 1
    xscaled = h/2*info.xn + x0 + h/2
    ws = info.w(xscaled)
    gs = info.g(xscaled)
    Dn = info.Dn
    y = 1j*ws
    delta = lambda r, y: -r/(2*(y + gs))
    R = lambda d: 2/h*Dn.dot(d) + d**2
    Ry = 1j*2*(1/h*Dn.dot(ws) + gs*ws)
    maxerr = max(np.abs(Ry))
    prev_err = np.inf
    o = 0 # Keep track of number of terms
    while maxerr > epsres:
        o += 1
        deltay = delta(Ry, y)
        y = y + deltay
        Ry = R(deltay)       
        maxerr = max(np.abs(Ry))
        if maxerr >= prev_err:
            success = 0
            break
        prev_err = maxerr
    if success == 1:
        pass
    du1 = y
    du2 = np.conj(du1)
    # LU
    u1 = h/2*scipy.linalg.lu_solve((info.DnLU, info.Dnpiv), du1, check_finite = False)
    u1 -= u1[-1]
    u2 = np.conj(u1)
    f1 = np.exp(u1)
    f2 = np.conj(f1)
    ap = (dy0 - y0*du2[-1])/(du1[-1] - du2[-1])   
    am = (dy0 - y0*du1[-1])/(du2[-1] - du1[-1])
    y1 = ap*f1 + am*f2
    dy1 = ap*du1*f1 + am*du2*f2
    phase = np.imag(u1[0])
    return y1[0], dy1[0], maxerr, success, phase


def osc_step(info, x0, h, y0, dy0, epsres = 1e-12, plotting = False, k = 0):
    """
    Advances the solution from x0 to x0+h, starting from the initial conditions
    y(x0) = y0, y'(x0) = dy0. It uses the Barnett series of order o (up to and
    including the (o)th correction), and the underlying functions w, g are
    represented on an (n+1)-node Chebyshev grid.
    ! Relies on info.wn and info.gn being populated with w(x0, ..., x0+h), etc.
    """
    success = 1
    xscaled = h/2*info.xn + x0 + h/2
    ws = info.wn
    gs = info.gn
    Dn = info.Dn
    y = 1j*ws
    delta = lambda r, y: -r/(2*(y + gs))
    R = lambda d: 2/h*Dn.dot(d) + d**2
    Ry = 1j*2*(1/h*Dn.dot(ws) + gs*ws)
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
                success = 0
                break
            prev_err = maxerr
        if success == 1:
            pass
    else:
        while o < k:
            o += 1
            deltay = delta(Ry, y)
            y = y + deltay
            Ry = R(deltay)       
            maxerr = max(np.abs(Ry))
    du1 = y
    du2 = np.conj(du1)
    # LU
    u1 = h/2*scipy.linalg.lu_solve((info.DnLU, info.Dnpiv), du1, check_finite = False)
    u1 -= u1[-1]
    u2 = np.conj(u1)
    f1 = np.exp(u1)
    f2 = np.conj(f1)
    ap = (dy0 - y0*du2[-1])/(du1[-1] - du2[-1])   
    am = (dy0 - y0*du1[-1])/(du2[-1] - du1[-1])
    y1 = ap*f1 + am*f2
    dy1 = ap*du1*f1 + am*du2*f2
    phase = np.imag(u1[0])
    if plotting:
        xplot = np.linspace(x0, x0+h, 500)
        L = interp(xscaled, xplot)
        u1plot = L.dot(u1)
        f1plot = np.exp(u1plot)
        f2plot = np.conj(f1plot)
        yplot = ap*f1plot + am*f2plot
        return xplot, yplot, xscaled, y1, maxerr
    else:
        return y1[0], dy1[0], maxerr, success, phase

def nonosc_step(info, x0, h, y0, dy0, epsres = 1e-12):

    """
    Advances the solution from x0 to x0+h, starting from the initial conditions
    y(x0) = y0, y'(x0) = dy0. It uses a Chebyshev spectral method with enough
    nodes to achieve epsres relative accuracy, starting from an initial n
    nodes.
    """
    success = 1 #TODO: this doesn't do anything in Cheb, can always add more nodes
    res = 0 #TODO: does nothing
    maxerr = 10*epsres
    N = info.nini
    Nmax = info.nmax
    yprev, dyprev, xprev = spectral_cheb(info, x0, h, y0, dy0, 0)
    while maxerr > epsres:
        N *= 2
        if N > Nmax:
            success = 0
            return 0, 0, maxerr, success, res
        y, dy, x = spectral_cheb(info, x0, h, y0, dy0, int(np.log2(N/info.nini))) 
        maxerr = np.abs((yprev[0] - y[0]))
        if np.isnan(maxerr):
            maxerr = np.inf
        yprev = y
        dyprev = dy
        xprev = x
    return y[0], dy[0], maxerr, success, res

def spectral_cheb(info, x0, h, y0, dy0, niter):
    """
    Utility function to apply a spectral method based on n Chebyshev nodes from
    x = x0 to x = x0+h, starting from the initial conditions y(x0) = y0, y'(x0)
    = dy0.
    """
    D, x = info.Ds[niter], info.nodes[niter]
    xscaled = h/2*x + x0 + h/2
    ws = info.w(xscaled)
    gs = info.g(xscaled)
    w2 = ws**2
    D2 = 4/h**2*D.dot(D) + 4/h*np.diag(gs).dot(D) + np.diag(w2)
    n = round(info.ns[niter])
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
    dy1 = 2/h*D.dot(y1)
    return y1, dy1, xscaled


class Solverinfo:
    """
    Class to store differentiation matrices, Chebyshev nodes, etc.
    """

    def __init__(self, w, g, h, nini, nmax, n, p):
        self.y = np.zeros(2, dtype = complex)
        self.w = w
        self.g = g
        self.h = h
        self.nini = nini
        self.nmax = nmax
        self.n = n
        self.p = p
        self.wn, self.gn = np.zeros(n + 1), np.zeros(n + 1)
        Dlength = int(np.log2(self.nmax/self.nini)) + 1
        self.Ds, self.nodes = [], []
        lognini = np.log2(self.nini)
        self.ns = np.logspace(lognini, lognini + Dlength - 1, num = Dlength, base = 2.0)#, dtype=int)
        for i in range(Dlength):
            D, x = cheb(self.nini*2**i)
            self.Ds.append(D)
            self.nodes.append(x)
        if self.n in self.ns:
            i = np.where(self.ns == self.n)[0][0]
            self.Dn, self.xn = self.Ds[i], self.nodes[i]
        else:
            self.Dn, self.xn = cheb(self.n)
        if self.p in self.ns:
            i = np.where(self.ns == self.p)[0][0]
            self.xp = self.nodes[i]
        else:
            self.xp = cheb(self.p)[1]
        self.xpinterp = np.cos(np.linspace(np.pi/(2*self.p), np.pi*(1 - 1/(2*self.p)), self.p))
        self.L = interp(self.xp, self.xpinterp)
        self.DnLU, self.Dnpiv = scipy.linalg.lu_factor(self.Dn, check_finite = False)


def setup(w, g, h0 = 1, nini = 16, nmax = 64, n = 16, p = 16):
    """
    Sets up the solver by generating differentiation matrices based on an
    increasing number of Chebyshev gridpoints: nini+1, 2*nini+1, ...,
    2*nmax+1).  Needs to be called before the first time the solver is ran or
    if nini or nmax are changed. 

    Parameters:
    -----------
    w: callable(t)
        Freqyency fynction in y'' + 2g(t)y' + w^2(t)y = 0.
    g: callable(t)
        Damping fynction in y'' + 2g(t)y' + w^2(t)y = 0.
    h0: float
        Initial stepsize.
    nini: int
        Lowest (number of nodes - 1) to start the Chebyshev spectral steps with.
    nmax: int
        Maximum (number of nodes - 1) for the spectral Chebyshev steps to
        try before decreasing the stepsize and reattempting the step with (nini + 1) nodes.
    n: int
        (Fixed) number of Chebyshev nodes to use for interpolation,
        integration, and differentiation in Riccati steps.
    p: int
        (Fixed) number of Chebyshev nodes to use for interpolation when
        determining the stepsize for Riccati steps.

    Returns:
    --------
    """
    info = Solverinfo(w, g, h0, nini, nmax, n, p)
    return info

def solve(info, xi, xf, yi, dyi, eps = 1e-12, epsh = 1e-12, xeval = [], hard_stop = False, t_eval = []):
    """
    Solves y'' + 2gy' + w^2y = 0 on the interval (xi, xf), starting from the
    initial conditions y(xi) = yi, y'(xi) = dyi. Keeps the residual of the ODE
    below eps, and returns an interpolated solution (dense output) at points
    xeval.

    Parameters:

    Returns:
    """
    w = info.w
    g = info.g
    Dn = info.Dn
    xn = info.xn
    n = info.n
    p = info.p
    hi = 0.1 # Initial stepsize for calculating derivatives
    
    # TODO: backwards integration
    xs = [xi]
    ys = [yi]
    dys = [dyi]
    phases = []
    steptypes = [0]
    successes = [1]
    y = yi
    dy = dyi
    yprev = y
    dyprev = dy
    wis = w(xi + hi/2 + hi/2*xn)
    gis = g(xi + hi/2 + hi/2*xn)
    wi = wis[-1]
    gi = gis[-1]
    dwi = 2/hi*Dn.dot(wis)[-1] 
    dgi = 2/hi*Dn.dot(gis)[-1] 
    # Choose initial stepsize
    hslo_ini = min(1e8, np.abs(1/wi))
    hosc_ini = min(1e8, np.abs(wi/dwi), np.abs(gi/dgi))
    hslo = choose_nonosc_stepsize(info, xi, hslo_ini)
    hosc = choose_osc_stepsize(info, xi, hosc_ini, epsh = epsh)  
    xcurrent = xi
    wnext = wi
    dwnext = dwi
    while abs(xcurrent - xf) > 1e-8 and xcurrent < xf:
        # Check how oscillatory the solution is
        #ty = np.abs(1/wnext)
        #tw = np.abs(wnext/dwnext)
        #tw_ty = tw/ty
        success = 0
        if hosc > hslo*5 and hosc*wnext/(2*np.pi) > 1:
            if hard_stop:
                if xcurrent + hosc > xf:
                    hosc = xf - xcurrent
                    xscaled = xcurrent + hosc/2 + hosc/2*xp 
                    wn = w(xscaled)
                    gn = g(xscaled) 
                if xcurrent + hslo > xf:
                    hslo = xf - xcurrent
            # Solution is oscillatory
            # Attempt osc step of size hosc
            y, dy, res, success, phase = osc_step(info, xcurrent, hosc, yprev, dyprev, epsres = eps)
            if success == 1:
                pass
            steptype = 1
        while success == 0:
            # Solution is not oscillatory, or previous step failed
            # Attempt Cheby step of size hslo
            y, dy, err, success, res = nonosc_step(info, xcurrent, hslo, yprev, dyprev, epsres = eps)
            phase = 0
            steptype = 0
            # If step still unsuccessful, halve stepsize
            if success == 0:
                hslo *= 0.5
            else:
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
        if steptype == 1:
            wnext = info.wn[0]
            gnext = info.gn[0]
            dwnext = 2/h*Dn.dot(info.wn)[0]
            dgnext = 2/h*Dn.dot(info.gn)[0]
        else:
            wnext = w(xcurrent + h)
            gnext = g(xcurrent + h)
            dwnext = 2/h*Dn.dot(w(xcurrent + h/2 + h/2*xn))[0]
            dgnext = 2/h*Dn.dot(g(xcurrent + h/2 + h/2*xn))[0]
        xcurrent += h
        if xcurrent < xf:
            hslo_ini = min(1e8, np.abs(1/wnext))
            hosc_ini = min(1e8, np.abs(wnext/dwnext), np.abs(gnext/dgnext))
            hosc = choose_osc_stepsize(info, xcurrent, hosc_ini, epsh = epsh)  
            hslo = choose_nonosc_stepsize(info, xcurrent, hslo_ini)
            yprev = y
            dyprev = dy
    return xs, ys, dys, successes, phases, steptypes

