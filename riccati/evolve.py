import numpy as np
import warnings
from riccati.chebutils import integrationm, interp
from riccati.stepsize import choose_osc_stepsize, choose_nonosc_stepsize
from riccati.step import osc_step, nonosc_step
import warnings

def osc_evolve(info, x0, x1, h, y0, epsres = 1e-12, epsh = 1e-12):
    """
    Allows continuous evolution between the independent variable values `x0`
    and `x1`. Starting from `x0` and `y = [y0, yd0]`, this function takes a
    Riccati step of size `h` or until `x1` is reached, and updates the
    attributes of `info` such that it can be called inside a `while` loop.
        
    Parameters
    ----------
    info: `Solverinfo` object
        `Solverinfo` object used to read off `info.Dn` for differentiation, and
        `info.wn`, `info.gn` for evaluations of w(x), g(x) over [x0, x0+h].
    x0: float
        Starting value of the independent variable.
    x1: float
        Maximum value of the independent variable. This may not be reached in a
        single step (i.e. a single call of `osc_evolve`), but will not be
        exceeded.
    h: float
        Stepsize. If the step would result in the independent variable
        exceeding `x1`, this will be adjusted.
    y0: np.ndarray [complex]
        Value of the state vector at `x0`.
    epsres: float
        Tolerance for the relative accuracy of Riccati steps.
    epsh: float
        Tolerance for choosing the stepsize for Riccati steps.
    
    Returns
    -------
    success: int
        0 if the step failed, 1 if successful.


    Warnings
    --------
    The user will need to set `info.wn`, `info.gn` correctly, as appropriate
    for a step of size `h`, before this function is called. This can be done by
    e.g. calling `choose_osc_stepsize`.
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
        xscaled = x0 + h/2 * (1 + info.xn)
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
        info.x = x0 + h
        # Determine new stepsize
        wnext = info.wn[0]
        gnext = info.gn[0]
        dwnext = 2/h*(info.Dn @ info.wn)[0]
        dgnext = 2/h*(info.Dn @ info.gn)[0]
        hosc_ini = sign*min(1e8, np.abs(wnext/dwnext), np.abs(gnext/dgnext))
        # Check if estimate for next stepsize would be out of range
        if sign*(info.x + hosc_ini) > sign*x1:
            hosc_ini = x1 - info.x
        info.h = choose_osc_stepsize(info, info.x, hosc_ini, epsh = epsh)  
    return success

def nonosc_evolve(info, x0, x1, h, y0, epsres = 1e-12, epsh = 0.2):
    """
    Allows continuous evolution between the independent variable values `x0`
    and `x1`. Starting from `x0` and `y = [y0, yd0]` takes a Chebyshev step of
    size `h` or until `x1` is reached, and updates the attributes of `info`
    such that it can be called inside a `while` loop.

    Parameters
    ----------
    info: `Solverinfo` object
        `Solverinfo` object used to read off `info.Dn` for differentiation, and
        `info.wn`, `info.gn` for evaluations of w(x), g(x) over [x0, x0+h].
    x0: float
        Starting value of the independent variable.
    x1: float
        Maximum value of the independent variable. This may not be reached in a
        single step (i.e. a single call of `osc_evolve`), but will not be
        exceeded.
    h: float
        Stepsize. If the step would result in the independent variable
        exceeding `x1`, this will be adjusted.
    y0: np.ndarray [complex]
        Value of the state vector at `x0`.
    epsres: float
        Tolerance for the relative accuracy of Riccati steps.
    epsh: float
        Tolerance for choosing the stepsize for Riccati steps.

    Notes
    -----
    Note that `epsh` is defined differently for Riccati and Chebyshev steps;
    the same value may therefore not be appropriate for both `osc_evolve()` and
    `nonosc_evolve()`.
    
    Returns
    -------
    success: int
        0 if the step failed, 1 if successful.


    Warnings
    --------
    The user will need to set `info.wn`, `info.gn` correctly, as appropriate
    for a step of size `h`, before this function is called. This can be done by
    e.g. calling `choose_osc_stepsize`.
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
        xscaled = x0 + h/2 * (1 + info.xn)
    info.h = h
    # Call nonoscillatory step
    y10, y11, maxerr, s = nonosc_step(info, x0, h, y0[0], y0[1], epsres = epsres)
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
        hslo_ini = sign*min(1e8, np.abs(1/wnext))
        # Check if estimate for next stepsize would be out of range
        if sign*(info.x + hslo_ini) > sign*x1:
            hslo_ini = x1 - info.x
        info.h = choose_nonosc_stepsize(info, info.x, hslo_ini, epsh = epsh)  
    return success

def solve(info, xi, xf, yi, dyi, eps = 1e-12, epsh = 1e-12, xeval = np.array([]), hard_stop = False, warn = False):
    """
    Solves y'' + 2gy' + w^2y = 0 on the interval (xi, xf), starting from the
    initial conditions y(xi) = yi, y'(xi) = dyi. Keeps the residual of the ODE
    below eps, and returns an interpolated solution (dense output) at points
    xeval.

    Parameters
    ----------
    info: Solverinfo object
        Objects containing differentiation matrices, etc.
    xi, xf: float
        Solution range.
    yi, dyi: complex
        Initial conditions, value of the dependent variable and its derivative at `xi`.
    eps: float
        Relative tolerance for the local error of both Riccati and Chebyshev type steps.
    epsh: float 
        Relative tolerance for choosing the stepsize of Riccati steps.
    xeval: list 
        List of x-values where the solution is to be interpolated (dense output) and returned.
    hard_stop: bool 
        Whether to force the solver to have a potentially smaller last
        stepsize, in order to stop exactly at `xf` (rather than allowing the
        solver to step over it and get the value of the solution by
        interpolation).
    warn: bool 
        Whether to display warnings, e.g. RuntimeWarnings, during a run. Due to
        the high level of adaptivity in this algorithm, it may throw several
        RuntimeWarnings even in a standard setup as it chooses the type of
        step, stepsize, and other parameters. For this reason, all warnings are
        silenced by default (`warn = False`). Set to `True` if you wish to see
        the warnings.

    Returns
    -------
    xs: list [float]
        Values of the independent variable at the internal steps of the solver.
    ys, dys: list [complex]
        Values of the dependent variable and its derivative at the internal steps of the solver.
    successes: list [int]
        Has elements 1 and 0: 1 denoting each successful step, 0 denoting unsuccessful steps. 
    phases:  list [complex]
        Complex phase of the solution accumulated during each successful Riccati step.
    steptypes: list [int]
        Types of successful steps taken: 1 for Riccati and 0 for Chebyshev. 
    yeval: numpy.array [complex]
        Dense output, i.e. values of the solution at the requested
        independent-variable values specified in `xeval`. If `xeval` was not
        given, then it is an empty numpy array of shape (0,).
    """
    if warn == False:
        warnings.simplefilter("ignore")
    else:
        warnings.simplefilter("default")

    w = info.w
    g = info.g
    Dn = info.Dn
    xn = info.xn
    n = info.n
    p = info.p
    hi = info.h0 # Initial stepsize for calculating derivatives
    
    # Is there dense output?
    info.denseout = False
    denselen = len(xeval)
    if denselen > 0:
        info.denseout = True
        info.intmat = integrationm(n+1)
        yeval = np.zeros(denselen, dtype = complex)
    else:
        yeval = np.empty(0)
    
    # Check if stepsize sign is consistent with direction of integration
    if (xf - xi)*hi < 0:
        warnings.warn("Direction of itegration does not match stepsize sign,\
                adjusting it so that integration happens from xi to xf.")
        hi *= -1

    # Determine direction
    intdir = np.sign(hi)

    # Warn if dense output points requested outside of interval
    if info.denseout:
        if intdir*xi < min(intdir*xeval) or intdir*xf > max(intdir*xeval):
            warnings.warn("Some dense output points lie outside the integration range!")

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
    wi = np.mean(wis)
    gi = np.mean(gis)
    dwi = np.mean(2/hi*(Dn @ wis))
    dgi = np.mean(2/hi*(Dn @ gis))
    # Choose initial stepsize
    hslo_ini = intdir*min(1e8, np.abs(1/wi))
    hosc_ini = intdir*min(1e8, np.abs(wi/dwi), np.abs(gi/dgi))
    # Check if we would be stepping over the integration range
    if hard_stop:
        if intdir*(xi + hosc_ini) > intdir*xf:
            hosc_ini = xf - xi
        if intdir*(xi + hslo_ini) > intdir*xf:
            hslo_ini = xf - xi
    hslo = choose_nonosc_stepsize(info, xi, hslo_ini)
    hosc = choose_osc_stepsize(info, xi, hosc_ini, epsh = epsh)  
    xcurrent = xi
    wnext = wi
    dwnext = dwi
    while abs(xcurrent - xf) > 1e-8 and intdir*xcurrent < intdir*xf:
        # Check how oscillatory the solution is
        #ty = np.abs(1/wnext)
        #tw = np.abs(wnext/dwnext)
        #tw_ty = tw/ty
        success = 0
        if intdir*hosc > intdir*hslo*5 and intdir*hosc*wnext/(2*np.pi) > 1:
            if hard_stop:
                if intdir*(xcurrent + hosc) > intdir*xf:
                    hosc = xf - xcurrent
                    xscaled = xcurrent + hosc/2 + hosc/2*info.xp 
                    wn = w(xscaled)
                    gn = g(xscaled)
                    info.wn = wn
                    info.gn = gn
                if intdir*(xcurrent + hslo) > intdir*xf:
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
            y, dy, err, success = nonosc_step(info, xcurrent, hslo, yprev, dyprev, epsres = eps)
            phase = 0
            steptype = 0
            # If step still unsuccessful, halve stepsize
            if success == 0:
                hslo *= 0.5
            else:
                pass
            if intdir*hslo < 1e-16:
                raise RuntimeError("Stepsize became too small, solution didn't converge with stepsize h > 1e-16")
        # Log step
        if steptype == 1:
            h = hosc
        else:
            h = hslo
        # If there were dense output points, check where:
        if info.denseout:
            positions = np.logical_or(np.logical_and(intdir*xeval >= intdir*xcurrent, intdir*xeval < intdir*(xcurrent+h)), np.logical_and(xeval == xf, xeval == xcurrent + h))
            xdense = xeval[positions] 
            if steptype == 1:
                #xscaled = xcurrent + h/2 + h/2*info.xn
                xscaled = 2/h*(xdense - xcurrent) - 1
                Linterp = interp(info.xn, xscaled)
                udense = Linterp @ info.un
                fdense = np.exp(udense)
                yeval[positions] = info.a[0]*fdense + info.a[1]*np.conj(fdense)
            else:
                xscaled = xcurrent + h/2 * (1 + info.nodes[1])
                Linterp = interp(xscaled[::-1], xdense)
                yeval[positions] = Linterp @ info.yn[::-1]
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
            dwnext = 2/h*(Dn @ info.wn)[0]
            dgnext = 2/h*(Dn @ info.gn)[0]
        else:
            wnext = w(xcurrent + h)
            gnext = g(xcurrent + h)
            dwnext = 2/h*(Dn @ w(xcurrent + h/2 * (1 + xn)))[0]
            dgnext = 2/h*(Dn @ g(xcurrent + h/2 * (1 + xn)))[0]
        xcurrent += h
        if intdir*xcurrent < intdir*xf:
            hslo_ini = intdir*min(1e8, np.abs(1/wnext))
            hosc_ini = intdir*min(1e8, np.abs(wnext/dwnext), np.abs(gnext/dgnext))
            if hard_stop:
                if intdir*(xcurrent + hosc_ini) > intdir*xf:
                    hosc_ini = xf - xcurrent
                if intdir*(xcurrent + hslo_ini) > intdir*xf:
                    hslo_ini = xf - xcurrent
            hosc = choose_osc_stepsize(info, xcurrent, hosc_ini, epsh = epsh)  
            hslo = choose_nonosc_stepsize(info, xcurrent, hslo_ini)
            yprev = y
            dyprev = dy
    return xs, ys, dys, successes, phases, steptypes, yeval

