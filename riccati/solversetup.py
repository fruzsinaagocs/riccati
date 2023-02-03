import numpy as np
from riccati.chebutils import cheb, interp, quadwts

class Solverinfo:
    """
    Class to store information of an ODE solve run. When initialized, it
    computes Chebyshev nodes and differentiation matrices neede for the run
    once and for all.

    Attributes
    ----------
    w, g: callable
        Frequency and friction function associated with the ODE, defined as :math:`\omega` and
        :math:`\gamma` in :math:`u'' + 2\gamma u' + \omega^2 u = 0`.
    h0: float
        Initial interval length which will be used to estimate the initial derivatives of w, g.
    nini, nmax: int
        Minimum and maximum (number of Chebyshev nodes - 1) to use inside
        Chebyshev collocation steps. The step will use `nmax` nodes or the
        minimum number of nodes necessary to achieve the required local error,
        whichever is smaller. If `nmax` > 2`nini`, collocation steps will be
        attempted with :math:`2^i` `nini` nodes at the ith iteration. 
    n: int
        (Number of Chebyshev nodes - 1) to use for computing Riccati steps.
    p: int
        (Number of Chebyshev nodes - 1) to use for estimating Riccati stepsizes.
    denseout: bool
        Defines whether or not dense output is required for the current run.
    h: float 
        Current stepsize.
    y: np.ndarray [complex]
        Current state vector of size (2,), containing the numerical solution and its derivative.
    wn, gn: np.ndarray [complex]
        The frequency and friction function evaluated at `n` + 1 Chebyshev nodes
        over the interval [x, x + `h` ], where x is the value of the independent
        variable at the start of the current step and `h` is the current
        stepsize.
    Ds: list [np.ndarray [float]]
       List containing differentiation matrices of sizes :math:`(2^i n_{\mathrm{ini}}
       + 1,2^i n_{\mathrm{ini}} + 1)` for :math:`i = 0, 1, \ldots, \lfloor \log_2\\frac{n_{\mathrm{max}}}{n_{\mathrm{ini}}} \\rfloor.`
    nodes: list [np.ndarray [float]]
        List containing vectors of Chebyshev nodes over the standard interval, ordered from +1 to -1, of sizes
        :math:`(2^i n_{\mathrm{ini}} + 1,)` for :math:`i = 0, 1, \ldots, \lfloor \log_2 \\frac{n_{\mathrm{max}}}{n_{\mathrm{ini}}} \\rfloor.`
    ns: np.ndarray [int] 
        Vector of lengths of node vectors stored in `nodes`, i.e. the integers
        :math:`2^i n_{\mathrm{ini}} + 1` for :math:`i = 0, 1, \ldots, \lfloor \log_2 \\frac{n_{\mathrm{max}}}{n_{\mathrm{ini}}} \\rfloor.`
    xn: np.ndarray [float] 
        Values of the independent variable evaluated at (`n` + 1) Chebyshev
        nodes over the interval [x, x + `h`], where x is the value of the
        independent variable at the start of the current step and `h` is the
        current stepsize.
    xp: np.ndarray [float]
        Values of the independent variable evaluated at (`p` + 1) Chebyshev
        nodes over the interval [x, x + `h`], where x is the value of the
        independent variable at the start of the current step and `h` is the
        current stepsize.
    xpinterp: np.ndarray [float]
        Values of the independent variable evaluated at `p` points 
        over the interval [x, x + `h`] lying in between Chebyshev nodes, where x is the value of the
        independent variable at the start of the current step and `h` is the
        current stepsize. The in-between points :math:`\\tilde{x}_p` are defined by
        
        .. math: \\tilde{x}_p = \cos\left( \\frac{(2k + 1)\pi}{2p} \\right), \quad k = 0, 1, \ldots p-1.

    L: np.ndarray [float]    
        Interpolation matrix of size (`p`+1, `p`), used for interpolating a
        function between the nodes `xp` and `xpinterp` (for computing Riccati
        stepsizes). 
    quadwts: np.ndarray [float]
        Vector of size (`n` + 1,) containing Clenshaw-Curtis quadrature weights.
    n_chebnodes: int
        Number of times Chebyhev nodes have been calculated, i.e.
        `riccati.chebutils.cheb` has been called.
    n_chebstep: int
        Number of Chebyshev steps attempted.
    n_chebits: int
        Number of times an iteration of the Chebyshev-grid-based
        collocation method has been performed (note that if `nmax` >=
        4`nini` then a single Chebyshev step may include multiple
        iterations!).
    n_LS:  int
        Number of times a linear system has been solved.
    n_riccstep: int
        Number of Riccati steps attempted.

    Methods
    -------
    increase:
        Increase the various counters (attributes starting with `n_`) by given values.
    output:
        Return the state of the counters as a dictionary.

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
        self.n_LS = 0
        self.n_riccstep = 0

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

    def increase(self, chebnodes = 0, chebstep = 0, chebits = 0,
                 LS = 0, riccstep = 0):
        """
        Increases the relevant attribute of the class (a counter for a specific
        arithmetic operation) by a given number. Used for generating performance statistics.

        Parameters
        ----------
        chebnodes: int
            Count by which to increase `riccati.solversetup.Solverinfo.n_chebnodes`.
        chebstep: int
            Count by which to increase `riccati.solversetup.Solverinfo.n_chebstep`.
        chebits: int
            Count by which to increase `riccati.solversetup.Solverinfo.n_chebits`.
        LS: int
            Count by which to increase `riccati.solversetup.Solverinfo.n_LS`.
        riccstep: int
            Count by which to increase `riccati.solversetup.Solverinfo.n_riccstep`.

        Returns
        -------
        None
        """
        self.n_chebnodes += chebnodes
        self.n_chebstep += chebstep
        self.n_chebits += chebits
        self.n_LS += LS
        self.n_riccstep += riccstep
    
    def output(self, steptypes):
        """
        Creates a dictionary of the counter-like attributes of `Solverinfo`,
        namely: the number of attempted Chebyshev steps `n_chebsteps`, the
        number of Chebyshev iterations `n_chebits`, the number of attempted
        Riccati steps `n_riccstep`, the number of linear solver `n_LS`, and the
        number of times Chebyshev nodes have been computed, `n_chebnodes`. It
        also logs the number of steps broken down by steptype: in the relevant
        fields, (n, m) means there were n attempted steps out of which m were
        successful.

        Parameters
        ----------
        steptypes: list [int]
            List of steptypes (of successful steps) produced by
            `riccati.evolve.solve()`, each element being 0 (Chebyshev step) or
            1 (Riccati step).

        Returns
        -------
        statdict: dict
            Dictionary with the following keywords:
            
            cheb steps: tuple [int] 
                (n, m), where n is the total number of attempted Chebyshev steps out of which m were successful.
            cheb iterations: int
                Total number of iterations of the Chebyshev collocation method. 
            ricc steps: tuple [int]
                (n, m), where n is the total number of attempted Chebyshev steps out of which m were successful.
            linear solves: int
                Total number of times a linear solve has been performed.
            cheb nodes: int
                Total number of times a call to compute Chebyshev nodes has been made.
        """
        statdict = {"cheb steps": (self.n_chebstep, sum(np.array(steptypes) == 0) - 1), 
                    "cheb iterations": self.n_chebits, 
                    "ricc steps": (self.n_riccstep, sum(np.array(steptypes) == 1)), 
                    "linear solves": self.n_LS, 
                    "cheb nodes": self.n_chebnodes}
        return statdict


def solversetup(w, g, h0 = 0.1, nini = 16, nmax = 32, n = 16, p = 16):
    """
    Sets up the solver by generating differentiation matrices based on an
    increasing number of Chebyshev gridpoints (see `riccati.solversetup.Solverinfo`).  Needs to be called
    before the first time the solver is ran or if nini or nmax are changed. 

    Parameters
    ----------
    w: callable(t)
        Frequency function in y'' + 2g(t)y' + w^2(t)y = 0.
    g: callable(t)
        Damping function in y'' + 2g(t)y' + w^2(t)y = 0.
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


