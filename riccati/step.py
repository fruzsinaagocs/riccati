import numpy as np
from riccati.chebutils import spectral_cheb

def nonosc_step(info, x0, h, y0, dy0, epsres = 1e-12):
    """
    A single Chebyshev step to be called from the `solve()` function.
    Advances the solution from `x0` by `h`, starting from the initial
    conditions `y(x0) = y0`, `y'(x0) = dy0`.
    The function uses a Chebyshev spectral method with an adaptive number of
    nodes. Initially, `info.nini` nodes are used, which is doubled in each
    iteration until `epsres` relative accuracy is reached or the number of
    nodes would exceed `info.nmax`. The relative error is measured as the
    difference between the predicted value of the dependent variable at the end
    of the step obtained in the current iteration and in the previous iteration
    (with half as many nodes). If the desired relative accuracy cannot be
    reached with `info.nmax` nodes, it is advised to decrease the stepsize `h`,
    increase `info.nmax`, or use a different approach. 

    Parameters
    ----------
    info: `Solverinfo` object
        `Solverinfo` object used to read off various matrices required for
        numerical differentiation, and `info.wn`, `info.gn` for evaluations of
        w(x), g(x) over [x0, x0+h].
    x0: float
        Starting value of the independent variable.
    h: float
        Stepsize.
    y0, dy0: complex
        Value of the dependent variable and its derivative at `x = x0`.
    epsres: float
        Tolerance for the relative accuracy of Chebyshev steps.

    Returns
    -------
    y[0], dy[0]: complex
        Value of the dependent variable and its derivative at the end of the
        step, `x = x0 + h`.
    maxerr: float
        (Absolute) value of the relative difference of the dependent variable
        at the end of the step as predicted in the last and the previous
        iteration.
    success: int
        Takes the value `1` if the asymptotic series has reached `epsres`
        residual, `0` otherwise.
    """
    success = 1
    maxerr = 10*epsres
    N = info.nini
    Nmax = info.nmax
    yprev, dyprev, xprev = spectral_cheb(info, x0, h, y0, dy0, 0)
    while np.abs((epsres*yprev[0] + epsres)/maxerr) < 1:
        N *= 2
        if N > Nmax:
            success = 0
            return 0, 0, maxerr, success
        y, dy, x = spectral_cheb(info, x0, h, y0, dy0, int(np.log2(N/info.nini))) 
        maxerr = yprev[0] - y[0]
        if np.isnan(maxerr):
            maxerr = np.inf
        yprev = y
        dyprev = dy
        xprev = x
    info.increase(chebstep = 1)
    if info.denseout:
        # Store interp points
        info.yn = y
        info.dyn = dy
    return y[0], dy[0], maxerr, success

def osc_step(info, x0, h, y0, dy0, epsres = 1e-12, plotting = False, k = 0):
    """
    A single Riccati step to be called from within the `solve()` function.
    Advances the solution from `x0` by `h`, starting from the initial conditions
    `y(x0) = y0`, `y'(x0) = dy0`. The function will increase the order of the
    asymptotic series used for the Riccati equation until a residual of
    `epsres` is reached or the residual stops decreasing. In the latter case,
    the asymptotic series cannot approximate the solution of the Riccati
    equation with the required accuracy over the given interval; the interval
    (`h`) should be reduced or another approximation should be used instead.

    Parameters
    ----------
    info: `Solverinfo` object
        `Solverinfo` object used to read off various matrices required for
        numerical differentiation, and `info.wn`, `info.gn` for evaluations of
        w(x), g(x) over [x0, x0+h].
    x0: float
        Starting value of the independent variable.
    h: float
        Stepsize.
    y0, dy0: complex
        Value of the dependent variable and its derivative at `x = x0`.
    epsres: float
        Tolerance for the relative accuracy of Riccati steps.

    Returns
    -------
    y1[0], dy1[0]: complex
        Value of the dependent variable and its derivative at the end of the
        step, `x = x0 + h`.
    maxerr: float
        Maximum value of the residual (after the final iteration of the
        asymptotic approximation) over the Chebyshev nodes across the interval.
    success: int
        Takes the value `1` if the asymptotic series has reached `epsres`
        residual, `0` otherwise.
    phase: complex
        Total phase change (not mod :math: `2\pi`!) of the dependent variable
        over the step.

    Warnings
    --------
    This function relies on `info.wn`, `info.gn` being set correctly, as
    appropriate for a step of size `h`. If `solve()` is calling this funciont,
    that is automatically taken care of, but otherwise needs to be done
    manually.
    """
    success = 1
    ws = info.wn
    gs = info.gn
    Dn = info.Dn
    y = 1j*ws
    delta = lambda r, y: -r/(2*(y + gs))
    R = lambda d: 2/h*(Dn @ d) + d**2
    Ry = 1j*2*(1/h*(Dn @ ws) + gs*ws)
    maxerr = max(np.abs(Ry))
    prev_err = np.inf
    if plotting == False:
        while maxerr > epsres:
            deltay = delta(Ry, y)
            y = y + deltay
            Ry = R(deltay)       
            maxerr = max(np.abs(Ry))
            if maxerr >= 2*prev_err:
                success = 0
                break
            prev_err = maxerr
    else:
        o = 0 # Keep track of number of terms
        while o < k:
            o += 1
            deltay = delta(Ry, y)
            y = y + deltay
            Ry = R(deltay)
            maxerr = max(np.abs(Ry))
    du1 = y
    du2 = np.conj(du1)
    if info.denseout:
        u1 = h/2*(info.intmat @ du1)
    else:
        u1 = h/2*(info.quadwts @ du1)
    u2 = np.conj(u1)
    f1 = np.exp(u1)
    f2 = np.conj(f1)
    ap = (dy0 - y0*du2[-1])/(du1[-1] - du2[-1])   
    am = (dy0 - y0*du1[-1])/(du2[-1] - du1[-1])
    y1 = ap*f1 + am*f2
    dy1 = ap*du1*f1 + am*du2*f2
    phase = np.imag(u1)
    info.increase(riccstep = 1)
    if info.denseout:
        info.un = u1
        info.a = (ap, am)
        y1 = y1[0]
    if plotting:
        return maxerr
    else:
        return y1, dy1[0], maxerr, success, phase


