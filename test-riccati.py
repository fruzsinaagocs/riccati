# Unit tests for riccati.py
import numpy as np
import riccati
import scipy.special as sp
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib import pyplot as plt

def test_cheb():
    a = 3.0
    f = lambda x: np.sin(a*x + 1.0)
    df = lambda x: a*np.cos(a*x + 1.0)
    for i in [16, 32]:
        D, x = riccati.cheb(i)
        maxerr = max(np.abs(np.matmul(D, f(x)) - df(x))) 
        print("At n = {} points, maximum error from Chebyshev differentiation is {}".format(i, maxerr))
        assert  maxerr < 1e-8

#def test_cheb_ab():
#    a = 3.0
#    f = lambda x: np.sin(a*x + 1.0)
#    df = lambda x: a*np.cos(a*x + 1.0)
#    x0 = 2.0
#    h = 1.0
#    for i in [16, 20, 30, 32, 64]:
#        D, x = riccati.cheb(i)
#        u = h/2*x + x0 + h/2
#        maxerr = max(np.abs(2/h*np.matmul(D, f(u)) - df(u))) 
#        print("At n = {} points, maximum error from Chebyshev differentiation is {}".format(i, maxerr))
#        assert  maxerr < 1e-8

def test_osc_step():
    w = lambda x: np.sqrt(x)
    x0 = 10.
    n_h = 1000; n_o = 21
    hs = np.linspace(1, 200, n_h)
    y0 = sp.airy(-x0)[0]
    dy0 = -sp.airy(-x0)[1]
    os = np.linspace(0, n_o-1, n_o, dtype=int)
    y_errs = np.zeros((n_h, n_o))
    ress = np.zeros((n_h, n_o))
    for i, h in enumerate(hs):
        y_ana = sp.airy(-(x0+h))[0]
        dy_ana = -sp.airy(-(x0+h))[1]
        for j, o in enumerate(os):
            y, dy, err = riccati.osc_step(w, x0, h, y0, dy0, o)
#            print("h = {}, o = {}, est res = {}, abs y err = {}, abs dy err = {}, rel y err = {}, rel dy err = {}".format(h, o, err, np.abs(y - y_ana), np. abs(dy - dy_ana), np.abs((y - y_ana)/y_ana), np.abs((dy - dy_ana)/dy_ana)))
            y_errs[i, j] = np.abs(y - y_ana)
            ress[i, j] = err

    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(15, 12))
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab20c.colors)
    ax[0].set_title("Error and residual in the Barnett series as a function of stepsize, Airy")
    ax[1].set_xlabel("$h$")
    ax[0].set_ylabel("Residual $R[y](x)$")
    ax[1].set_ylabel("Absolute error $\Delta y$")
    for o in os:
        ax[0].loglog(hs, ress[:, o], label='o = {}'.format(o), color='C{}'.format(o))
        ax[1].loglog(hs, y_errs[:, o], color='C{}'.format(o))
    ax[0].legend()
    plt.tight_layout()
    plt.show()

    # For a given choice of h, o
    # y_err = np.abs(y - y_ana)
    # assert y_err < 1e-8 




