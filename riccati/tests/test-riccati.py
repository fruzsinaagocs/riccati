# Unit tests for riccati.py
import numpy as np
import riccati
import scipy.special as sp
import math
import mpmath
import time
import matplotlib.pyplot as plt

#def test_integration():
#    n = 16
#    a = 3.0
#    f = lambda x: np.sin(a*x + 1.0)
#    df = lambda x: a*np.cos(a*x + 1.0)
#    D, x = riccati.cheb(n)
#    dfs = df(x)
#    fs = f(x)
#    fs -= fs[-1]
#    Im = riccati.integrationm(n+1)
#    fs_est = Im @ dfs
#    maxerr = max(np.abs((fs_est - fs)/fs))
#    assert maxerr < 1e-8
#
#def test_denseoutput():
#    w = lambda x: np.sqrt(x)
#    g = lambda x: np.zeros_like(x)
#    info = riccati.setup(w, g, n = 32, p = 32)
#    xi = 1e0
#    xf = 1e6
#    eps = 1e-12
#    epsh = 1e-13
#    yi = sp.airy(-xi)[0] + 1j*sp.airy(-xi)[2]
#    dyi = -sp.airy(-xi)[1] - 1j*sp.airy(-xi)[3]
#    Neval = int(1e4)
#    xeval = np.linspace(xi, xf, Neval) 
#    startd = time.time_ns()
#    xs, ys, dys, ss, ps, stypes, yeval = riccati.solve(info, xi, xf, yi, dyi, xeval = xeval, eps = eps, epsh = epsh)
#    start = time.time_ns() 
#    ytrue = np.array([mpmath.airyai(-x) + 1j*mpmath.airybi(-x) for x in xeval]) # This is a little slow, but accurate
#    yerr = np.abs((ytrue - yeval)/ytrue)
#    maxerr = max(yerr)
#    assert maxerr < 1e-6
#
#def test_quadwts():
#    a = 3.0
#    f = lambda x: np.sin(a*x + 1.0)
#    df = lambda x: a*np.cos(a*x + 1.0)
#    for i in [16, 32]:
#        D, x = riccati.cheb(i)
#        w = riccati.quadwts(i)
#        maxerr = w.dot(df(x)) - (f(1) - f(-1)) 
#        assert  maxerr < 1e-8
#
#def test_cheb():
#    a = 3.0
#    f = lambda x: np.sin(a*x + 1.0)
#    df = lambda x: a*np.cos(a*x + 1.0)
#    for i in [16, 32]:
#        D, x = riccati.cheb(i)
#        maxerr = max(np.abs(np.matmul(D, f(x)) - df(x))) 
#        assert  maxerr < 1e-8
#
#def test_osc_step():
#    w = lambda x: np.sqrt(x) 
#    g = lambda x: np.zeros_like(x)
#    info = riccati.setup(w, g)
#    x0 = 10.0
#    h = 20.0
#    epsres = 1e-12
#    xscaled = x0 + h/2 + h/2*info.xn
#    info.wn = w(xscaled)
#    info.gn = g(xscaled)
#    y0 = sp.airy(-x0)[0]
#    dy0 = -sp.airy(-x0)[1]
#    y_ana = sp.airy(-(x0+h))[0]
#    dy_ana = -sp.airy(-(x0+h))[1]
#    y, dy, res, success, phase = riccati.osc_step(info, x0, h, y0, dy0, epsres = epsres)
#    y_err = np.abs((y - y_ana)/y_ana)
#    assert y_err < 1e-8 and res < epsres
#
#def test_nonosc_step():
#    w = lambda x: np.sqrt(x) 
#    g = lambda x: np.zeros_like(x)
#    info = riccati.setup(w, g)
#    x0 = 1.0
#    h = 0.5
#    eps = 1e-12
#    y0 = sp.airy(-x0)[2]
#    dy0 = - sp.airy(-x0)[3]
#    y, dy, err, success = riccati.nonosc_step(info, x0, h, y0, dy0, epsres = eps)
#    y_ana = sp.airy(-x0-h)[2]
#    dy_ana = -sp.airy(-x0-h)[3]
#    y_err = np.abs((y - y_ana)/y_ana)
#    dy_err = np.abs((dy - dy_ana)/dy_ana)
#    assert y_err < 1e-8 and dy_err < 1e-8
#
#
###def test_choose_stepsize():
###    w = lambda x: np.sqrt(x)
###    g = lambda x: 0.0
###    x0 = 1e8
###    h = 1e10
###    hnew = riccati.choose_stepsize(w, g, x0, h, p = 16)
###    print(hnew)
###    # TODO: Not quite sure how to test this...
#
#def test_solve_airy():
#    w = lambda x: np.sqrt(x)
#    g = lambda x: np.zeros_like(x)
#    info = riccati.setup(w, g, n = 32, p = 32)
#    xi = 1e0
#    xf = 1e6
#    eps = 1e-12
#    epsh = 1e-13
#    yi = sp.airy(-xi)[0] + 1j*sp.airy(-xi)[2]
#    dyi = -sp.airy(-xi)[1] - 1j*sp.airy(-xi)[3]
#    xs, ys, dys, ss, ps, stypes = riccati.solve(info, xi, xf, yi, dyi, eps = eps, epsh = epsh)
#    xs = np.array(xs)
#    ys = np.array(ys)
#    ytrue = np.array([mpmath.airyai(-x) + 1j*mpmath.airybi(-x) for x in xs])
#    yerr = np.abs((ytrue - ys)/ytrue)
#    maxerr = max(yerr)
#    print(xs, ys, yerr, maxerr)
#    assert maxerr < 1e-6

def test_solve_airy_backwards():
    w = lambda x: np.sqrt(x)
    g = lambda x: np.zeros_like(x)
    info = riccati.setup(w, g, n = 32, p = 32, h0 = 1000)
    xi = 1e2
    xf = 1e0
    eps = 1e-12
    epsh = 1e-13
    yi = sp.airy(-xi)[0] + 1j*sp.airy(-xi)[2]
    dyi = -sp.airy(-xi)[1] - 1j*sp.airy(-xi)[3]
    xs, ys, dys, ss, ps, stypes = riccati.solve(info, xi, xf, yi, dyi, eps = eps, epsh = epsh, hard_stop = True)
    xs = np.array(xs)
    ys = np.array(ys)
    ytrue = np.array([mpmath.airyai(-x) + 1j*mpmath.airybi(-x) for x in xs])
    yerr = np.abs((ytrue - ys)/ytrue)
    maxerr = max(yerr)
    print(xs, ys, yerr, maxerr)
    assert maxerr < 1e-6



#def test_solve_burst():
#    m = int(1e6) # Frequency parameter
#    w = lambda x: np.sqrt(m**2 - 1)/(1 + x**2)
#    g = lambda x: np.zeros_like(x)
#    bursty = lambda x: np.sqrt(1 + x**2)/m*(np.cos(m*np.arctan(x)) + 1j*np.sin(m*np.arctan(x))) 
#    burstdy = lambda x: 1/np.sqrt(1 + x**2)/m*((x + 1j*m)*np.cos(m*np.arctan(x))\
#            + (-m + 1j*x)*np.sin(m*np.arctan(x)))
#    xi = -m
#    xf = m
#    yi = bursty(xi)
#    dyi = burstdy(xi)
#    eps = 1e-10
#    epsh = 1e-12
#    info = riccati.setup(w, g, n = 32, p = 32)
#    xs, ys, dys, ss, ps, types = riccati.solve(info, xi, xf, yi, dyi, eps = eps, epsh = epsh)
#    xs = np.array(xs)
#    ys = np.array(ys)
#    ytrue = bursty(xs)
#    yerr = np.abs((ytrue - ys))/np.abs(ytrue)
#    maxerr = max(yerr)
#    assert maxerr < 1e-7
#    
#def test_osc_evolve():
#    w = lambda x: np.sqrt(x)
#    g = lambda x: np.zeros_like(x)
#    info = riccati.setup(w, g, n = 32, p = 32)
#    xi = 1e2
#    xf = 1e6
#    eps = 1e-12
#    epsh = 1e-13
#    yi = sp.airy(-xi)[0] + 1j*sp.airy(-xi)[2]
#    dyi = -sp.airy(-xi)[1] - 1j*sp.airy(-xi)[3]
#    # Store things
#    xs, ys, dys = [], [], []
#    # Always necessary for setting info.y
#    info.y = np.array([yi, dyi])
#    # Always necessary for setting info.wn, info.gn, and for getting an initial stepsize
#    hi = 2*xi
#    hi = riccati.choose_osc_stepsize(info, xi, hi, epsh = epsh)
#    info.h = hi
#    # Not necessary here because info.x is already xi, but in general it might be:
#    info.x = xi
#    while info.x < xf:
#        status = riccati.osc_evolve(info, info.x, xf, info.h, info.y, epsres = eps, epsh = epsh)
#        if status != 1:
#            break
#        xs.append(info.x)
#        ys.append(info.y[0])
#    xs = np.array(xs)
#    ys = np.array(ys)
#    ytrue = np.array([mpmath.airyai(-x) + 1j*mpmath.airybi(-x) for x in xs])
#    yerr = np.abs((ytrue - ys)/ytrue)
#    maxerr = max(yerr)
#    assert maxerr < 1e-6
#   
#def test_nonosc_evolve():
#    w = lambda x: np.sqrt(x)
#    g = lambda x: np.zeros_like(x)
#    info = riccati.setup(w, g, n = 32, p = 32)
#    xi = 1e0
#    xf = 4e1
#    eps = 1e-12
#    epsh = 2e-1 # Note different definition of epsh for Chebyshev steps!
#    yi = sp.airy(-xi)[0] + 1j*sp.airy(-xi)[2]
#    dyi = -sp.airy(-xi)[1] - 1j*sp.airy(-xi)[3]
#    # Store things
#    xs, ys, dys = [], [], []
#    # Always necessary for setting info.y
#    info.y = np.array([yi, dyi])
#    # Necessary for getting an initial stepsize
#    hi = 1/w(xi)
#    hi = riccati.choose_nonosc_stepsize(info, xi, hi, epsh = epsh)
#    info.h = hi
#    # Not necessary here because info.x is already xi, but in general it might be:
#    info.x = xi
#    while info.x < xf:
#        status = riccati.nonosc_evolve(info, info.x, xf, info.h, info.y, epsres = eps, epsh = epsh)
#        if status != 1:
#            break
#        xs.append(info.x)
#        ys.append(info.y[0])
#    xs = np.array(xs)
#    ys = np.array(ys)
#    ytrue = np.array([mpmath.airyai(-x) + 1j*mpmath.airybi(-x) for x in xs])
#    yerr = np.abs((ytrue - ys)/ytrue)
#    maxerr = max(yerr)
#    assert maxerr < 1e-10

