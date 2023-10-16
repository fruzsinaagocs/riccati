import numpy as np
from riccati.chebutils import cheb, integrationm, quadwts
from riccati.solversetup import solversetup
from riccati.evolve import solve, osc_evolve, nonosc_evolve
from riccati.step import osc_step, nonosc_step
from riccati.stepsize import choose_osc_stepsize, choose_nonosc_stepsize
import scipy.special as sp
import mpmath
import warnings
import pytest

def test_integration():
    n = 16
    a = 3.0
    f = lambda x: np.sin(a*x + 1.0)
    df = lambda x: a*np.cos(a*x + 1.0)
    D, x = cheb(n)
    dfs = df(x)
    fs = f(x)
    fs -= fs[-1]
    Im = integrationm(n+1)
    fs_est = Im @ dfs
    maxerr = max(np.abs((fs_est - fs)/fs))
    assert maxerr < 1e-8

def test_denseoutput():
    w = lambda x: np.sqrt(x)
    g = lambda x: np.zeros_like(x)
    info = solversetup(w, g, n = 32, p = 32)
    xi = 1e0
    xf = 1e6
    eps = 1e-12
    epsh = 1e-13
    yi = sp.airy(-xi)[0] + 1j*sp.airy(-xi)[2]
    dyi = -sp.airy(-xi)[1] - 1j*sp.airy(-xi)[3]
    Neval = int(1e2)
    xeval = np.linspace(xi, xf, Neval) 
    xs, ys, dys, ss, ps, stypes, yeval = solve(info, xi, xf, yi, dyi,\
                                                       xeval = xeval,\
                                                       eps = eps, epsh = epsh)
    ytrue = np.array([mpmath.airyai(-x) + 1j*mpmath.airybi(-x) for x in xeval])
    yerr = np.abs((ytrue - yeval)/ytrue)
    maxerr = max(yerr)
    assert maxerr < 1e-6

def test_denseoutput_xbac():
    w = lambda x: np.sqrt(x)
    g = lambda x: np.zeros_like(x)
    info = solversetup(w, g, n = 32, p = 32)
    xi = 1e0
    xf = 1e6
    eps = 1e-12
    epsh = 1e-13
    yi = sp.airy(-xi)[0] + 1j*sp.airy(-xi)[2]
    dyi = -sp.airy(-xi)[1] - 1j*sp.airy(-xi)[3]
    Neval = int(1e2)
    xeval = np.linspace(xf, xi, Neval) 
    xs, ys, dys, ss, ps, stypes, yeval = solve(info, xi, xf, yi, dyi,\
                                                       xeval = xeval,\
                                                       eps = eps, epsh = epsh)
    ytrue = np.array([mpmath.airyai(-x) + 1j*mpmath.airybi(-x) for x in xeval])
    yerr = np.abs((ytrue - yeval)/ytrue)
    maxerr = max(yerr)
    assert maxerr < 1e-6

def test_denseoutput_warn():
    w = lambda x: np.sqrt(x)
    g = lambda x: np.zeros_like(x)
    info = solversetup(w, g, n = 32, p = 32)
    xi = 1e0
    xf = 1e6
    eps = 1e-12
    epsh = 1e-13
    yi = sp.airy(-xi)[0] + 1j*sp.airy(-xi)[2]
    dyi = -sp.airy(-xi)[1] - 1j*sp.airy(-xi)[3]
    Neval = int(1e2)
    xeval = np.linspace(xi-10.0, xi, Neval) 
    # Turn on warnings
    warnings.simplefilter("always")
    with warnings.catch_warnings(record = True) as w:
        xs, ys, dys, ss, ps, stypes, yeval = solve(info, xi, xf, yi, dyi,\
                                                           xeval = xeval,\
                                                           eps = eps, epsh = epsh, warn = True)
        assert "outside the integration range" in str(w[0].message) 

def test_quadwts():
    a = 3.0
    f = lambda x: np.sin(a*x + 1.0)
    df = lambda x: a*np.cos(a*x + 1.0)
    for i in [16, 32]:
        D, x = cheb(i)
        w = quadwts(i)
        maxerr = w.dot(df(x)) - (f(1) - f(-1)) 
        assert  maxerr < 1e-8

def test_cheb():
    a = 3.0
    f = lambda x: np.sin(a*x + 1.0)
    df = lambda x: a*np.cos(a*x + 1.0)
    for i in [16, 32]:
        D, x = cheb(i)
        maxerr = max(np.abs(np.matmul(D, f(x)) - df(x))) 
        assert  maxerr < 1e-8

def test_osc_step():
    w = lambda x: np.sqrt(x) 
    g = lambda x: np.zeros_like(x)
    info = solversetup(w, g)
    x0 = 10.0
    h = 20.0
    epsres = 1e-12
    xscaled = x0 + h/2 + h/2*info.xn
    info.wn = w(xscaled)
    info.gn = g(xscaled)
    y0 = sp.airy(-x0)[0]
    dy0 = -sp.airy(-x0)[1]
    y_ana = sp.airy(-(x0+h))[0]
    dy_ana = -sp.airy(-(x0+h))[1]
    y, dy, res, success, phase = osc_step(info, x0, h, y0, dy0, epsres = epsres)
    y_err = np.abs((y - y_ana)/y_ana)
    assert y_err < 1e-8 and res < epsres

def test_nonosc_step():
    w = lambda x: np.sqrt(x) 
    g = lambda x: np.zeros_like(x)
    info = solversetup(w, g)
    x0 = 1.0
    h = 0.5
    eps = 1e-12
    y0 = sp.airy(-x0)[2]
    dy0 = - sp.airy(-x0)[3]
    y, dy, err, success = nonosc_step(info, x0, h, y0, dy0, epsres = eps)
    y_ana = sp.airy(-x0-h)[2]
    dy_ana = -sp.airy(-x0-h)[3]
    y_err = np.abs((y - y_ana)/y_ana)
    dy_err = np.abs((dy - dy_ana)/dy_ana)
    assert y_err < 1e-8 and dy_err < 1e-8

def test_solve_airy():
    w = lambda x: np.sqrt(x)
    g = lambda x: np.zeros_like(x)
    info = solversetup(w, g, n = 32, p = 32)
    xi = 1e0
    xf = 1e6
    eps = 1e-12
    epsh = 1e-13
    yi = sp.airy(-xi)[0] + 1j*sp.airy(-xi)[2]
    dyi = -sp.airy(-xi)[1] - 1j*sp.airy(-xi)[3]
    xs, ys, dys, ss, ps, stypes = solve(info, xi, xf, yi, dyi, eps = eps, epsh = epsh)
    xs = np.array(xs)
    ys = np.array(ys)
    ytrue = np.array([mpmath.airyai(-x) + 1j*mpmath.airybi(-x) for x in xs])
    yerr = np.abs((ytrue - ys)/ytrue)
    maxerr = max(yerr)
    print(maxerr)
    assert maxerr < 1e-6

def test_solve_airy_backwards():
    w = lambda x: np.sqrt(x)
    g = lambda x: np.zeros_like(x)
    info = solversetup(w, g, n = 32, p = 32)
    xi = 1e6
    xf = 1e0
    eps = 1e-12
    epsh = 1e-13
    yi = sp.airy(-xi)[0] + 1j*sp.airy(-xi)[2]
    dyi = -sp.airy(-xi)[1] - 1j*sp.airy(-xi)[3]
    xs, ys, dys, ss, ps, stypes = solve(info, xi, xf, yi, dyi,\
                                                eps = eps, epsh = epsh,\
                                                hard_stop = True)
    xs = np.array(xs)
    ys = np.array(ys)
    ytrue = np.array([mpmath.airyai(-x) + 1j*mpmath.airybi(-x) for x in xs])
    yerr = np.abs((ytrue - ys)/ytrue)
    maxerr = max(yerr)
    print(maxerr, stypes)
    assert maxerr < 1e-6

def test_denseoutput_backwards_xfor():
    w = lambda x: np.sqrt(x)
    g = lambda x: np.zeros_like(x)
    info = solversetup(w, g, n = 32, p = 32)
    xi = 1e6
    xf = 1e0
    eps = 1e-12
    epsh = 1e-13
    yi = sp.airy(-xi)[0] + 1j*sp.airy(-xi)[2]
    dyi = -sp.airy(-xi)[1] - 1j*sp.airy(-xi)[3]
    Neval = int(1e2)
    xeval = np.linspace(xi, xf, Neval) 
    xs, ys, dys, ss, ps, stypes, yeval = solve(info, xi, xf, yi, dyi,\
                                                       xeval = xeval,\
                                                       eps = eps, epsh = epsh,\
                                                       hard_stop = True)
    ytrue = np.array([mpmath.airyai(-x) + 1j*mpmath.airybi(-x) for x in xeval])
    yerr = np.abs((ytrue - yeval)/ytrue)
    maxerr = max(yerr)
    print(maxerr)
    assert maxerr < 1e-6

def test_denseoutput_backwards_xback():
    w = lambda x: np.sqrt(x)
    g = lambda x: np.zeros_like(x)
    info = solversetup(w, g, n = 32, p = 32)
    xi = 1e6
    xf = 1e0
    eps = 1e-12
    epsh = 1e-13
    yi = sp.airy(-xi)[0] + 1j*sp.airy(-xi)[2]
    dyi = -sp.airy(-xi)[1] - 1j*sp.airy(-xi)[3]
    Neval = int(1e2)
    xeval = np.linspace(xf, xi, Neval) 
    xs, ys, dys, ss, ps, stypes, yeval = solve(info, xi, xf, yi, dyi,\
                                                       xeval = xeval,\
                                                       eps = eps, epsh = epsh,\
                                                       hard_stop = True)
    ytrue = np.array([mpmath.airyai(-x) + 1j*mpmath.airybi(-x) for x in xeval])
    yerr = np.abs((ytrue - yeval)/ytrue)
    maxerr = max(yerr)
    print(maxerr)
    assert maxerr < 1e-6

def test_solve_burst():
    m = int(1e6) # Frequency parameter
    w = lambda x: np.sqrt(m**2 - 1)/(1 + x**2)
    g = lambda x: np.zeros_like(x)
    bursty = lambda x: np.sqrt(1 + x**2)/m*(np.cos(m*np.arctan(x)) + 1j*np.sin(m*np.arctan(x))) 
    burstdy = lambda x: 1/np.sqrt(1 + x**2)/m*((x + 1j*m)*np.cos(m*np.arctan(x))\
            + (-m + 1j*x)*np.sin(m*np.arctan(x)))
    xi = -m
    xf = m
    yi = bursty(xi)
    dyi = burstdy(xi)
    eps = 1e-10
    epsh = 1e-12
    info = solversetup(w, g, n = 32, p = 32)
    xs, ys, dys, ss, ps, types = solve(info, xi, xf, yi, dyi, eps = eps, epsh = epsh)
    xs = np.array(xs)
    ys = np.array(ys)
    ytrue = bursty(xs)
    yerr = np.abs((ytrue - ys))/np.abs(ytrue)
    maxerr = max(yerr)
    assert maxerr < 2e-7
    
def test_osc_evolve():
    w = lambda x: np.sqrt(x)
    g = lambda x: np.zeros_like(x)
    info = solversetup(w, g, n = 32, p = 32)
    xi = 1e2
    xf = 1e6
    eps = 1e-12
    epsh = 1e-13
    yi = sp.airy(-xi)[0] + 1j*sp.airy(-xi)[2]
    dyi = -sp.airy(-xi)[1] - 1j*sp.airy(-xi)[3]
    # Store things
    xs, ys, dys = [], [], []
    # Always necessary for setting info.y
    info.y = np.array([yi, dyi])
    # Always necessary for setting info.wn, info.gn, and for getting an initial stepsize
    hi = 2*xi
    hi = choose_osc_stepsize(info, xi, hi, epsh = epsh)
    info.h = hi
    # Not necessary here because info.x is already xi, but in general it might be:
    info.x = xi
    while info.x < xf:
        status = osc_evolve(info, info.x, xf, info.h, info.y, epsres = eps, epsh = epsh)
        if status != 1:
            break
        xs.append(info.x)
        ys.append(info.y[0])
    xs = np.array(xs)
    ys = np.array(ys)
    ytrue = np.array([mpmath.airyai(-x) + 1j*mpmath.airybi(-x) for x in xs])
    yerr = np.abs((ytrue - ys)/ytrue)
    maxerr = max(yerr)
    print("Forward osc evolve max error:", maxerr)
    assert maxerr < 1e-6
   
def test_nonosc_evolve():
    w = lambda x: np.sqrt(x)
    g = lambda x: np.zeros_like(x)
    info = solversetup(w, g, n = 32, p = 32)
    xi = 1e0
    xf = 4e1
    eps = 1e-12
    epsh = 2e-1 # Note different definition of epsh for Chebyshev steps!
    yi = sp.airy(-xi)[0] + 1j*sp.airy(-xi)[2]
    dyi = -sp.airy(-xi)[1] - 1j*sp.airy(-xi)[3]
    # Store things
    xs, ys, dys = [], [], []
    # Always necessary for setting info.y
    info.y = np.array([yi, dyi])
    # Necessary for getting an initial stepsize
    hi = 1/w(xi)
    hi = choose_nonosc_stepsize(info, xi, hi, epsh = epsh)
    info.h = hi
    # Not necessary here because info.x is already xi, but in general it might be:
    info.x = xi
    while info.x < xf:
        status = nonosc_evolve(info, info.x, xf, info.h, info.y, epsres = eps, epsh = epsh)
        if status != 1:
            break
        xs.append(info.x)
        ys.append(info.y[0])
    xs = np.array(xs)
    ys = np.array(ys)
    ytrue = np.array([mpmath.airyai(-x) + 1j*mpmath.airybi(-x) for x in xs])
    yerr = np.abs((ytrue - ys)/ytrue)
    maxerr = max(yerr)
    print("Forward nonosc evolve max error:", maxerr)
    assert maxerr < 1e-10

def test_osc_evolve_backwards():
    w = lambda x: np.sqrt(x)
    g = lambda x: np.zeros_like(x)
    info = solversetup(w, g, n = 32, p = 32)
    xi = 1e6
    xf = 1e2
    eps = 1e-12
    epsh = 1e-13
    yi = sp.airy(-xi)[0] + 1j*sp.airy(-xi)[2]
    dyi = -sp.airy(-xi)[1] - 1j*sp.airy(-xi)[3]
    # Store things
    xs, ys, dys = [], [], []
    # Always necessary for setting info.y
    info.y = np.array([yi, dyi])
    # Always necessary for setting info.wn, info.gn, and for getting an initial stepsize
    hi = -xi/10
    hi = choose_osc_stepsize(info, xi, hi, epsh = epsh)
    info.h = hi
    # Not necessary here because info.x is already xi, but in general it might be:
    info.x = xi
    while info.x > xf:
        status = osc_evolve(info, info.x, xf, info.h, info.y, epsres = eps, epsh = epsh)
        if status != 1:
            break
        xs.append(info.x)
        ys.append(info.y[0])
    xs = np.array(xs)
    ys = np.array(ys)
    ytrue = np.array([mpmath.airyai(-x) + 1j*mpmath.airybi(-x) for x in xs])
    yerr = np.abs((ytrue - ys)/ytrue)
    maxerr = max(yerr)
    print("Backwards osc evolve max error:", maxerr)
    assert maxerr < 1e-6
   
def test_nonosc_evolve_backwards():
    w = lambda x: np.sqrt(x)
    g = lambda x: np.zeros_like(x)
    info = solversetup(w, g, n = 32, p = 32)
    xi = 4e1
    xf = 1e0
    eps = 1e-12
    epsh = 2e-1 # Note different definition of epsh for Chebyshev steps!
    yi = sp.airy(-xi)[0] + 1j*sp.airy(-xi)[2]
    dyi = -sp.airy(-xi)[1] - 1j*sp.airy(-xi)[3]
    # Store things
    xs, ys, dys = [], [], []
    # Always necessary for setting info.y
    info.y = np.array([yi, dyi])
    # Necessary for getting an initial stepsize
    hi = -1/w(xi)
    hi = choose_nonosc_stepsize(info, xi, hi, epsh = epsh)
    info.h = hi
    # Not necessary here because info.x is already xi, but in general it might be:
    info.x = xi
    while info.x > xf:
        status = nonosc_evolve(info, info.x, xf, info.h, info.y, epsres = eps, epsh = epsh)
        if status != 1:
            break
        xs.append(info.x)
        ys.append(info.y[0])
    xs = np.array(xs)
    ys = np.array(ys)
    ytrue = np.array([mpmath.airyai(-x) + 1j*mpmath.airybi(-x) for x in xs])
    yerr = np.abs((ytrue - ys)/ytrue)
    maxerr = max(yerr)
    print("Backward nonosc evolve max error:", maxerr)
    assert maxerr < 1e-10

def test_bremer237():
           
    nruns = 1
    # Integration range
    ti = -1.0
    tf =  1.0
    for jj in range(0,21):
       dnu = 2.0**jj
       w = lambda t: np.sqrt( dnu**2*(1-t**2*np.cos(3*t)))
       g = lambda t: 0*t
       # Initial conditions
       ui  = 1
       dui = dnu
       info   = riccati.solversetup(w, g)
       for ii in range(nruns):
         ts, ys, *misc = riccati.solve(info, ti, tf, ui, dui, hard_stop=True)
    return 

def test_legendre():

    def legepol(n,x):
      if n==0:
        return 1.0
      if n==1:
        return x;
      p1 = 1
      p2 = x
      p  = 0
      for j in range(2,n+1):
        p  = ((2.0*j-1.0)*x*p2-(j-1.0)*p1)/j
        p1 = p2
        p2 = p
      return p
    
    # return the value and derivative of the Legendre polynomial of degree n at 0
    def lege0(n):
       dnu = n
       if n % 2 == 0 :
         x1    = sp.gammaln(0.5+dnu/2.0)-sp.gammaln(1.0+dnu/2.0)
         val0  = np.exp(x1)/np.sqrt(np.pi)
         if n % 4 == 2:
           val0=-val0
         
         der0 = 0
       else:
         x1   = -sp.gammaln(1.5+dnu/2) + sp.gammaln(dnu/2)
         val0 = 0
         der0 = dnu*(dnu+1)*1/np.sqrt(np.pi)*1/2 * np.exp(x1)
         if n % 4 == 3:
           der0=-der0
       return np.array([val0,der0])
    
    # Integration range
    ti     = 0.0
    tf     = 0.9
    nruns  = 1
    m      = 10000
    eps    = 1.0e-12
    for jj in range(0,21):
       norder = 2**jj
       dnu = norder
       y0= lege0(norder)
       w = lambda t: np.sqrt(dnu*(dnu+1)/(1.0-t**2) )
       g = lambda t: -t/(1.0-t**2)
        
       # Initial conditions
       ui  = y0[0]
       dui = y0[1]
       info   = riccati.solversetup(w, g)   
       for ii in range(nruns):
         ts, ys, *misc, y_eval  = riccati.solve(info, ti, tf, ui, dui, eps=eps, hard_stop=True)    
       errmax  = 0
       l = len(ys)
       val0 = ys[l-1]
       val = legepol(norder,tf)
       errmax=abs(val-val0)
       print(norder,errmax)
    return

