import numpy as np

def choose_nonosc_stepsize(info, x0, h, epsh = 0.2):
    """
    Chooses the stepsize for spectral Chebyshev steps, based on the variation
    of 1/w, the approximate timescale over which the solution changes. If over
    the suggested interval h 1/w changes by a fraction of :math:`\pm``epsh` or more, the interval is
    halved, otherwise it's accepted.

    Parameters
    ----------
    info: Solverinfo object
        Solverinfo object which is used to retrieve Solverinfo.xp, the
        (p+1) Chebyshev nodes used for interpolation to determine the stepsize.
    x0: float
        Current value of the independent variable.
    h: float
        Initial estimate of the stepsize.
    epsh: float
        Tolerance parameter defining how much 1/w(x) is allowed to change over
        the course of the step.

    Returns
    -------
    h: float
        Refined stepsize over which 1/w(x) does not change by more than epsh/w(x).
    """
    xscaled = x0 + h/2 * (1 + info.xp)
    ws = info.w(xscaled)
    if np.isnan(ws.sum()) == True or max(np.abs(ws)) > (1 + epsh)/abs(h):
         return choose_nonosc_stepsize(info, x0, h/2, epsh = epsh)
    else:
        return h

def choose_osc_stepsize(info, x0, h, epsh = 1e-12):
    """
    Chooses the stepsize `h` over which the functions w(x), g(x) can be
    represented sufficiently accurately. Evaluations of w(x) and g(x) at (p+1)
    Chebyshev nodes (given by `info.xp`) between [`x0`, `x0+h`] are used to infer
    w(x), g(x) values at p points halfway between the (p+1) nodes on the half
    circle, 

    .. math:: x_{p,\mathrm{interp}} = \cos\left( \\frac{\pi(2n+1)}{2p} \\right), \quad n = 0, 1, \ldots, p-1.

    These target nodes given by `info.xpinterp`. The same (p+1) points
    will then be used to compute a Riccati step.  The interpolated values are
    then compared to the actual values of w(x), g(x) at `info.xpinterp`. If the
    largest relative error in w, g exceeds `epsh`, `h` is halved.

    Parameters
    ----------
    info: Solverinfo object
        `Solverinfo` object used for two purposes: `info.xpinterp`,
        `info.xp` are looked up to determine the source and target points
        to determine whether Chebyshev interpolation of w(x), g(x) is accurate
        enough over the step; and `info.wn`, `info.gn` may be used to read off the
        w, g values at the source points.
    x0: float
        Current value of the independent variable.
    h: float
        Initial estimate of the stepsize.
    epsh: float
        Tolerance parameter defining the maximum relative error Chebyshev
        interpolation of w, g is allowed to have over the course of the
        proposed step [`x0`, `x0+h`].

    Returns
    -------
    h: float
        Refined stepsize over which Chebyshev interpolation of w, g has a
        relative error no larger than epsh.
    """
    w, g, L = info.w, info.g, info.L
    t = x0 + h/2 * (1 + info.xpinterp)
    s = x0 + h/2 * (1 + info.xp)
    if info.p == info.n:
        info.wn = w(s)
        info.gn = g(s)
        ws = info.wn
        gs = info.gn
    else:
        info.wn = w(x0 + h/2 * (1 + info.xn))
        info.gn = g(x0 + h/2 * (1 + info.xn))
        ws = w(s)
        gs = g(s)
    wana = w(t)
    west = L @ ws
    gana = g(t)
    gest = L @ gs
    maxwerr = max(np.abs((west - wana)/wana))
    maxgerr = max(np.abs((gest - gana)/gana))
    maxerr = max(maxwerr, maxgerr)
    if maxerr > epsh:
        return choose_osc_stepsize(info, x0, h*min(0.7, 0.9*(epsh/maxerr)**(1/(info.p-1))), epsh = epsh)
    else:
        return h
    #TODO: what if h is too small to begin with?


