import numpy as np
from riccati.chebutils import cheb, interp, quadwts

class Solverinfo:
    """
    Class to store differentiation matrices, Chebyshev nodes, etc.
    """

    def __init__(self, w, g, h, nini, nmax, n, p):

        # Parameters
        self.w = w
        self.g = g
        self.h0 = h
        self.nini = nini
        self.nmax = nmax
        self.n = n
        self.p = p
        self.denseout = False

        # Run statistics
        self.n_chebnodes = 0
        self.n_chebstep = 0
        self.n_chebits = 0
        self.n_LS2x2 = 0
        self.n_LS = 0
        self.n_riccstep = 0
        self.n_sub = 0
        self.n_LU = 0

        self.h = self.h0
        self.y = np.zeros(2, dtype = complex)
        self.wn, self.gn = np.zeros(n + 1), np.zeros(n + 1)
        Dlength = int(np.log2(self.nmax/self.nini)) + 1
        self.Ds, self.nodes = [], []
        lognini = np.log2(self.nini)
        self.ns = np.logspace(lognini, lognini + Dlength - 1, num = Dlength, base = 2.0)#, dtype=int)

        for i in range(Dlength):
            D, x = cheb(self.nini*2**i)
            self.increase(chebnodes = 1)
            self.Ds.append(D)
            self.nodes.append(x)
        if self.n in self.ns:
            i = np.where(self.ns == self.n)[0][0]
            self.Dn, self.xn = self.Ds[i], self.nodes[i]
        else:
            self.Dn, self.xn = cheb(self.n)
            self.increase(chebnodes = 1)
        if self.p in self.ns:
            i = np.where(self.ns == self.p)[0][0]
            self.xp = self.nodes[i]
        else:
            self.xp = cheb(self.p)[1]
            self.increase(chebnodes = 1)
        self.xpinterp = np.cos(np.linspace(np.pi/(2*self.p), np.pi*(1 - 1/(2*self.p)), self.p))
        self.L = interp(self.xp, self.xpinterp)
        self.increase(LS = 1)
        self.quadwts = quadwts(n)
        self.increase(LU = 1)

    def increase(self, chebnodes = 0, chebstep = 0, chebits = 0, LS2x2 = 0, LS = 0, riccstep = 0, sub = 0, LU = 0):
        self.n_chebnodes += chebnodes
        self.n_chebstep += chebstep
        self.n_chebits += chebits
        self.n_LS2x2 += LS2x2
        self.n_LS += LS
        self.n_riccstep += riccstep
        self.n_LU += LU
        self.n_sub += sub
    
    def output(self, steptypes):
        statdict = {"cheb steps": (self.n_chebstep, sum(np.array(steptypes) == 0) - 1), 
                    "cheb iterations": self.n_chebits, 
                    "ricc steps": (self.n_riccstep, sum(np.array(steptypes) == 1)), 
                    "linear solves": self.n_LS, 
                    "linear solves 2x2": self.n_LS2x2, 
                    "cheb nodes": self.n_chebnodes,
                    "LU decomp": self.n_LU, 
                    "substitution": self.n_sub}
        return statdict


def solversetup(w, g, h0 = 0.1, nini = 16, nmax = 32, n = 16, p = 16):
    """
    Sets up the solver by generating differentiation matrices based on an
    increasing number of Chebyshev gridpoints: nini+1, 2*nini+1, ...,
    2*nmax+1).  Needs to be called before the first time the solver is ran or
    if nini or nmax are changed. 

    Parameters
    ----------
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

    Returns
    -------
    info: Solverinfo object
        Solverinfo object created with attributes set as per the input parameters.
    """
    info = Solverinfo(w, g, h0, nini, nmax, n, p)
    return info


