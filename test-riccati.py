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
#    x0 = 10.0
#    h = 30.0
#    epsres = 1e-12
#    y0 = sp.airy(-x0)[0]
#    dy0 = -sp.airy(-x0)[1]
#    y_ana = sp.airy(-(x0+h))[0]
#    dy_ana = -sp.airy(-(x0+h))[1]
#    y, dy, res = riccati.osc_step(w, x0, h, y0, dy0, epsres = epsres)
#    y_err = np.abs((y - y_ana)/y_ana)
#    assert y_err < 1e-8 and res < epsres
#
#def test_choose_stepsize():
#    w = lambda x: np.sqrt(x)
#    x0 = 1e8
#    h = 1e10
#    hnew = riccati.choose_stepsize(w, x0, h, p = 16)
#    print(hnew)
#    # TODO: Not quite sure how to test this...

def test_solve():
    w = lambda x: np.sqrt(x)
    g = lambda x: 0
    xi = 1e2
    xf = 1e8
    eps = 1e-10
    epsh = 1e-12
    yi = sp.airy(-xi)[0]
    dyi = -sp.airy(-xi)[1]
    xs, ys, dys, ss = riccati.solve(w, g, xi, xf, yi, dyi, eps = eps, epsh = epsh)
    xs = np.array(xs)
    ys = np.array(ys)
    ss = np.array(ss)

    xcts = np.logspace(np.log10(xi), np.log10(xf), 1000)
    ycts = np.array([sp.airy(-x)[0] for x in xcts])
    ytrue = np.array([mpmath.airyai(-x) for x in xs])
    yerr = np.abs((ytrue - ys)/ytrue)
    hs = np.array([xs[i] - xs[i-1] for i in range(1, xs.shape[0]) ])
    print(ys)
    print(ytrue)
    print(yerr)

    fig, ax = plt.subplots(2, 1, sharex = True)
    #ax[0].semilogx(xcts, ycts, color='black', label= 'True solution')
    #ax[0].semilogx(xs, ys, '.', color='C1', label = 'numerical solution')
    #ax[0].set_ylabel('Solution, $y(x)$')
    ax[0].set_ylabel('Stepsize, $h(x)$')
    ax[0].loglog(xs[:-1], hs)
    ax[1].loglog(xs[ss==True], yerr[ss==True], '.-', color='C0', label='Converged')
    ax[1].loglog(xs[ss==False], yerr[ss==False], '.', color='C1', label='Diverged before $\epsilon$ was hit')
    ax[1].loglog(xs, eps*np.ones_like(xs), color='black')
    ax[1].set_ylabel('Relative error, $|\Delta y/y|$')
    ax[1].set_xlabel('$x$') 
    ax[0].set_title('Numerical solution of the Airy equation, $\epsilon = ${}, $\epsilon_h = ${}'.format(eps, epsh))
    ax[1].legend()
    plt.tight_layout()
    plt.show()




