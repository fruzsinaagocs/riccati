# Unit tests for riccati.py
import numpy as np
import riccati
import scipy.special as sp
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib import pyplot as plt
import math
import mpmath

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
#    x0 = 10.0
#    h = 30.0
#    epsres = 1e-12
#    y0 = sp.airy(-x0)[0]
#    dy0 = -sp.airy(-x0)[1]
#    y_ana = sp.airy(-(x0+h))[0]
#    dy_ana = -sp.airy(-(x0+h))[1]
#    y, dy, res, success, phase = riccati.osc_step(w, g, x0, h, y0, dy0, epsres = epsres)
#    y_err = np.abs((y - y_ana)/y_ana)
#    assert y_err < 1e-8 and res < epsres

#def test_nonosc_step_singleh_singlen():
#    w = lambda x: np.sqrt(x) 
#    g = lambda x: np.zeros_like(x)
#    x0 = 1.0
#    h = 1e-3
#    n = 32
#    x = riccati.cheb(n)[-1]
#    x = h/2*x + x0 + h/2
#    y0 = sp.airy(-x0)[2]
#    dy0 = - sp.airy(-x0)[3]
#    y, dy, err, success, res = riccati.nonosc_step(w, g, x0, h, y0, dy0, n=n)
#    y_ana = sp.airy(-x)[2]
#    dy_ana = -sp.airy(-x)[3]
#    y_err = np.abs((y - y_ana)/y_ana)
#    #print('residuals from lsqst: ', ress)
#    #print(min(ress), max(ress))
#    #mini = np.argmin(ress)
#    #print(hs[mini], errs[mini])
#    fig, ax = plt.subplots(2, 1, sharex=True)
#    ax[0].plot(x, y, label='numerical solution')
#    ax[0].plot(x, y_ana, label='actual solution')
#    ax[0].legend()
#    ax[1].semilogy(x, y_err)
#    plt.show()
#    #assert y_err < 1e-8 and res < epsres

#def test_nonosc_step_singlen():
#    w = lambda x: np.sqrt(x) 
#    g = lambda x: np.zeros_like(x)
#    n = 16
#    x0 = 0.0
#    hs = np.logspace(-5, 1, 1000)
#    errs = np.zeros_like(hs)
#    ys = np.zeros(hs.shape, dtype=complex)
#    dyanas = np.zeros_like(ys)
#    yanas = np.zeros_like(ys)
#    y0 = sp.airy(-x0)[2]
#    dy0 = - sp.airy(-x0)[3]
#    for i, h in enumerate(hs):
#        y, dy, err, success, res = riccati.nonosc_step(w, g, x0, h, y0, dy0, n=n)
#        y_ana = sp.airy(-(x0+h))[2]
#        dy_ana = - sp.airy(-(x0+h))[3]
#        y_err = np.abs((y - y_ana)/y_ana)
#        errs[i] = y_err[0]
#        ys[i] = y[0]
#        dyanas[i] = dy_ana
#        yanas[i] = y_ana
#    fig, ax = plt.subplots(2, 1, sharex=True)
#    ax[0].semilogx(x0 + hs, ys, label='numerical solution')
#    ax[0].semilogx(x0 + hs, yanas, label='actual solution')
#    ax[0].set_ylabel("$y(x)$")
#    ax[0].legend()
#    ax[1].loglog(x0 + hs, errs)
#    ax[1].loglog(x0 + hs, 2*(x0+hs)/np.abs(yanas/dyanas))
#    ax[1].set_xlabel("$x$")
#    ax[1].set_ylabel("Relative error")
#    ax[0].set_title("Numerical solution of the Airy equation with spectral method of order $n=${}".format(n))
#    plt.show()
#    #assert y_err < 1e-8 and res < epsres
#
#def test_nonosc_step_singleh():
#    w = lambda x: np.sqrt(x) 
#    g = lambda x: np.zeros_like(x)
#    h = 1.
#    x0 = 0.0
#    ns = range(2, 32)
#    errs = np.zeros(len(ns))
#    ys = np.zeros(len(ns), dtype=complex)
#    yanas = np.zeros_like(ys)
#    y0 = sp.airy(-x0)[2]
#    dy0 = - sp.airy(-x0)[3]
#    for i, n in enumerate(ns):
#        y, dy, err, success, res = riccati.nonosc_step(w, g, x0, h, y0, dy0, n=n)
#        y_ana = sp.airy(-(x0+h))[2]
#        y_err = np.abs((y - y_ana)/y_ana)
#        errs[i] = y_err[0]
#    fig, ax = plt.subplots(1, 1, sharex=True)
#    ax.semilogy(ns, errs)
#    ax.legend()
#    ax.set_xlabel("Number of Chebyshev nodes, $n$")
#    ax.set_ylabel("Relative error")
#    plt.title("Numerical solution of Airy equation with spectral method with $n$ nodes")
#    plt.show()
#    #assert y_err < 1e-8 and res < epsres
#
#def test_nonosc_step():
#    w = lambda x: np.sqrt(x) 
#    g = lambda x: np.zeros_like(x)
#    x0 = 0.0
#    hs = np.logspace(-3, 1, 500)
#    ns = range(2, 33, 4)
#    ys = np.zeros((hs.shape[0], len(ns)), dtype=complex)
#    errs = np.zeros_like(ys)
#    yanas = np.zeros_like(hs)
#    dyanas = np.zeros_like(hs)
#    y0 = sp.airy(-x0)[2]
#    dy0 = - sp.airy(-x0)[3]
#    for i, h in enumerate(hs):
#        for j, n in enumerate(ns):
#            y, dy, err, success, res = riccati.nonosc_step(w, g, x0, h, y0, dy0, n=n)
#            y_ana = sp.airy(-(x0+h))[2]
#            dy_ana = -sp.airy(-(x0+h))[3]
#            y_err = np.abs((y - y_ana)/y_ana)
#            errs[i, j] = y_err[0]
#            ys[i, j] = y[0]
#            yanas[i] = y_ana
#            dyanas[i] = dy_ana
#    fig, ax = plt.subplots(1, 1, sharex=True)
#    for j, n in enumerate(ns):
#        ax.loglog(hs, errs[:, j], label="$n = ${}".format(n))
#    ax.loglog(hs, 2*(x0+hs)/np.abs(yanas/dyanas), label="$\\tau_x$")
#    ax.legend()
#    ax.set_xlabel("stepsize $h$")
#    ax.set_ylabel("relative error")
#    plt.title("Numerical solution of Airy equation with spectral method with $n$ nodes")
#    plt.show()
#    #assert y_err < 1e-8 and res < epsres

#def test_choose_stepsize():
#    w = lambda x: np.sqrt(x)
#    g = lambda x: 0.0
#    x0 = 1e8
#    h = 1e10
#    hnew = riccati.choose_stepsize(w, g, x0, h, p = 16)
#    print(hnew)
#    # TODO: Not quite sure how to test this...

#def test_solve_airy():
#    w = lambda x: np.sqrt(x)
#    g = lambda x: np.zeros_like(x)
#    xi = 1e2
#    xf = 1e4
#    eps = 1e-12
#    epsh = 1e-13
#    yi = sp.airy(-xi)[0] + 1j*sp.airy(-xi)[2]
#    dyi = -sp.airy(-xi)[1] - 1j*sp.airy(-xi)[3]
#    xs, ys, dys, ss, ps = riccati.solve(w, g, xi, xf, yi, dyi, eps = eps, epsh = epsh)
#    xs = np.array(xs)
#    ys = np.array(ys)
#    ytrue = np.array([mpmath.airyai(-x) + 1j*mpmath.airybi(-x) for x in xs])
#    yerr = np.abs((ytrue - ys)/ytrue)
#    maxerr = max(yerr)
#    assert maxerr < 1e-8

#def test_solve_airy_nonosc():
#    w = lambda x: np.sqrt(x)
#    g = lambda x: np.zeros_like(x)
#    xi = 1e-3
#    xf = 1e3
#    eps = 1e-11
#    epsh = 1e-13
#    yi = sp.airy(-xi)[0] #+ 1j*sp.airy(-xi)[2]
#    dyi = -sp.airy(-xi)[1] #- 1j*sp.airy(-xi)[3]
#    xs, ys, dys, ss, ps, types = riccati.solve(w, g, xi, xf, yi, dyi, eps = eps, epsh = epsh)
#    xs = np.array(xs)
#    ys = np.array(ys)
#    types = np.array(types)
#    ytrue = np.array([mpmath.airyai(-x) for x in xs]) #+ 1j*mpmath.airybi(-x) for x in xs])
#    ncts = 1000
#    xcts = np.logspace(np.log10(xi), np.log10(xf), ncts)
#    ycts = np.array([sp.airy(-x)[0] for x in xcts]) 
#    yerr = np.abs((ytrue - ys)/ytrue)
#    maxerr = max(yerr)
#    
#    fig, ax = plt.subplots(2, 1, sharex=True)
#    ax[0].plot(xcts, ycts, color='black', lw=1.0, label='True solution')
#    ax[0].plot(xs[types==0], ys[types==0], '.', color='C1', label='Nonosc step')
#    ax[0].plot(xs[types==1], ys[types==1], '.', color='C0', label='Osc step')
#    ax[1].semilogy(xs, yerr, color='black')
#    ax[0].set_ylabel("$y(x)$")
#    ax[1].set_ylabel("Relative error")
#    ax[1].set_xlabel("$x$")
#    ax[0].legend()
#    plt.show()
#    #assert maxerr < 1e-8

def test_solve_burst():
    m = int(1e8) # Frequency parameter
    w = lambda x: np.sqrt(m**2 - 1)/(1 + x**2)
    g = lambda x: np.zeros_like(x)
    bursty = lambda x: np.sqrt(1 + x**2)/m*(np.cos(m*np.arctan(x)) + 1j*np.sin(m*np.arctan(x))) 
    burstdy = lambda x: 1/np.sqrt(1 + x**2)/m*((x + 1j*m)*np.cos(m*np.arctan(x))\
            + (-m + 1j*x)*np.sin(m*np.arctan(x)))
    xi = -m
    xf = m
    yi = bursty(xi)
    dyi = burstdy(xi)
#    print(yi, dyi)
    eps = 1e-4
    epsh = 1e-12
    xs, ys, dys, ss, ps, types = riccati.solve(w, g, xi, xf, yi, dyi, eps = eps, epsh = epsh)
    xs = np.array(xs)
    ys = np.array(ys)
    types = np.array(types)
    ytrue = bursty(xs)
    ncts = 5000
    if xi > 0: 
        xcts = np.logspace(np.log10(xi), np.log10(xf), ncts)
    else:
        xcts = np.linspace(xi, xf, ncts)
    ycts = bursty(xcts)
    yerr = np.abs((ytrue - ys))/np.abs(ytrue)
    maxerr = max(yerr)
    
    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(xcts, ycts, color='black', lw=1.0, label='True solution')
    ax[0].plot(xs[types==0], ys[types==0], '.', color='C1', label='Nonosc step')
    ax[0].plot(xs[types==1], ys[types==1], '.', color='C0', label='Osc step')
    ax[1].semilogy(xs, yerr, '.-', color='black')
    ax[0].set_ylabel("$y(x)$")
    ax[1].set_ylabel("Relative error")
    ax[1].set_xlabel("$x$")
    ax[0].legend()
    plt.show()


