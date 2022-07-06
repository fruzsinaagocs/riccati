# Unit tests for riccati.py
import numpy as np
import riccati
import scipy.special as sp
import matplotlib
#matplotlib.use("Qt5Agg")
from matplotlib import pyplot as plt
import math
import mpmath

def test_cheb():
    a = 3.0
    f = lambda x: np.sin(a*x + 1.0)
    df = lambda x: a*np.cos(a*x + 1.0)
    for i in [16, 32]:
        D, x = riccati.cheb(i)
        maxerr = max(np.abs(np.matmul(D, f(x)) - df(x))) 
        assert  maxerr < 1e-8

def test_osc_step():
    w = lambda x: np.sqrt(x) 
    g = lambda x: np.zeros_like(x)
    x0 = 10.0
    h = 30.0
    epsres = 1e-12
    y0 = sp.airy(-x0)[0]
    dy0 = -sp.airy(-x0)[1]
    y_ana = sp.airy(-(x0+h))[0]
    dy_ana = -sp.airy(-(x0+h))[1]
    y, dy, res, success, phase = riccati.osc_step(w, g, x0, h, y0, dy0, epsres = epsres)
    y_err = np.abs((y - y_ana)/y_ana)
    assert y_err < 1e-8 and res < epsres

def test_nonosc_step_single():
    w = lambda x: np.sqrt(x) 
    g = lambda x: np.zeros_like(x)
    x0 = 1.0
    h = 0.5
    eps = 1e-12
    y0 = sp.airy(-x0)[2]
    dy0 = - sp.airy(-x0)[3]
    y, dy, err, success, res = riccati.nonosc_step(w, g, x0, h, y0, dy0, epsres = eps)
    y_ana = sp.airy(-x0-h)[2]
    dy_ana = -sp.airy(-x0-h)[3]
    y_err = np.abs((y - y_ana)/y_ana)
    dy_err = np.abs((dy - dy_ana)/dy_ana)
    assert y_err < 1e-8 and dy_err < 1e-8


#def test_choose_stepsize():
#    w = lambda x: np.sqrt(x)
#    g = lambda x: 0.0
#    x0 = 1e8
#    h = 1e10
#    hnew = riccati.choose_stepsize(w, g, x0, h, p = 16)
#    print(hnew)
#    # TODO: Not quite sure how to test this...

def test_solve_airy():
    w = lambda x: np.sqrt(x)
    g = lambda x: np.zeros_like(x)
    xi = 1e0
    xf = 1e6
    eps = 1e-12
    epsh = 1e-13
    yi = sp.airy(-xi)[0] + 1j*sp.airy(-xi)[2]
    dyi = -sp.airy(-xi)[1] - 1j*sp.airy(-xi)[3]
    xs, ys, dys, ss, ps, stypes, statdict = riccati.solve(w, g, xi, xf, yi, dyi, eps = eps, epsh = epsh)
    xs = np.array(xs)
    ys = np.array(ys)
    ytrue = np.array([mpmath.airyai(-x) + 1j*mpmath.airybi(-x) for x in xs])
    yerr = np.abs((ytrue - ys)/ytrue)
    maxerr = max(yerr)
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
    eps = 1e-8
    epsh = 1e-12
    xs, ys, dys, ss, ps, types, statdict = riccati.solve(w, g, xi, xf, yi, dyi, eps = eps, epsh = epsh, n = 32)
    xs = np.array(xs)
    ys = np.array(ys)
    ytrue = bursty(xs)
    yerr = np.abs((ytrue - ys))/np.abs(ytrue)
    maxerr = max(yerr)
    assert maxerr < 1e-7
    

