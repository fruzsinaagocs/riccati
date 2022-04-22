# Unit tests for riccati.py
import numpy as np
import riccati
import scipy.special as sp
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib import pyplot as plt
import math
import mpmath

#def test_num_diff():
#    x0 = 10.0
#    n_h = 200 
#    n_o = 5
#    N = 16 # Cheb nodes
#    hs = np.logspace(-5, np.log10(200), n_h)
#    os = np.linspace(1, n_o, n_o, dtype=int)
#    abserrs = np.zeros((n_h, n_o))
#    relerrs = np.zeros((n_h, n_o))
#    f = lambda x: np.sqrt(x)
#    df = lambda n, x: (-1)**(n-1)*math.prod([2*i+1 for i in range(n-1)])/2**n*x**(-(2*n-1)/2)
#    for i, h in enumerate(hs):
#        D, x = riccati.cheb(N)
#        u = h/2*x + x0 + h/2
#        df_num = f(u)
#        for j, o in enumerate(os):
#            df_num = 2/h*np.matmul(D, df_num)
#            df_ana = df(o, u)
#            abserrs[i, j] = max(np.abs(df_num - df_ana))
#            relerrs[i, j] = max(np.abs((df_num - df_ana)/df_ana))
#     
#    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(5, 6))
#    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab20c.colors)
#    ax[0].set_title("Error in numerical differentiation based on Cheb nodes, $f(x) = \sqrt{x}$, $n = 16$, $x_0 = 1$", wrap=True)
#    ax[1].set_xlabel("Stepsize $h$")
#    ax[0].set_ylabel("Absolute error $|\Delta y|$")
#    ax[1].set_ylabel("Relative error $|\Delta y/y|$")
#    for o in os:
#        ax[0].loglog(hs, abserrs[:, o-1], label='o = {}'.format(o), color='C{}'.format(o))
#        ax[1].loglog(hs, relerrs[:, o-1], color='C{}'.format(o))
#    ax[0].legend()
#    plt.tight_layout()
#    #plt.savefig("num_diff_airy_vary_o.pdf")
#    plt.show()
#
#def test_num_diff2():
#    x0 = 1e2
#    n_h = 200
#    n_n = 5
#    o = 1 # order of derivative
#    hs = np.logspace(-5, np.log10(10000), n_h)
#    ns = np.linspace(6, 4*n_n+5, n_n, dtype=int)
#    abserrs = np.zeros((n_h, n_n))
#    relerrs = np.zeros((n_h, n_n))
#    f = lambda x: np.sqrt(x)
#    df = lambda n, x: (-1)**(n-1)*math.prod([2*i+1 for i in range(n-1)])/2**n*x**(-(2*n-1)/2)
#    for i, h in enumerate(hs):
#        for j, n in enumerate(ns):
#            D, x = riccati.cheb(n)
#            u = h/2*x + x0 + h/2
#            #print("nodes: ", u)
#            df_num = f(u)
#            #print("f(nodes): ", df_num)
#            for k in range(o):
#                df_num = 2/h*np.matmul(D, df_num)
#                #print("f^({})(nodes): ".format(k+1), df_num)
#            df_ana = df(o, u)
#            #print("At h = {}, n = {}, df_num = {}, df_ana = {}".format(h, n, df_num, df_ana))
#            abserrs[i, j] = max(np.abs((df_num - df_ana)))
#            relerrs[i, j] = max(np.abs((df_num - df_ana)/df_ana))
#     
#    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(5, 6))
#    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab20c.colors)
#    ax[0].set_title("Error in numerical differentiation based on Cheb nodes, $f(x) = \sqrt{x}$, $x_0 = 100$, $o = 3$", wrap=True)
#    ax[1].set_xlabel("$h$")
#    ax[0].set_ylabel("Absolute error $|\Delta y|$")
#    ax[1].set_ylabel("Relative error $|\Delta y/y|$")
#    for i, n in enumerate(ns):
#        ax[0].loglog(hs, abserrs[:, i], label='n = {}'.format(n), color='C{}'.format(i))
#        ax[1].loglog(hs, relerrs[:, i], color='C{}'.format(i))       
#    ax[0].legend()
#    plt.tight_layout()
#    #plt.savefig("num_diff_airy_vary_n.pdf")
#    plt.show()
#    #maxerr = max(np.abs(np.matmul(D, f(x)) - df(x))) 
#
#
#def test_num_diff3():
#    n_x = 6
#    x0s = np.logspace(1, n_x, n_x, dtype=int)
#    n_h = 200
#    n = 32
#    o = 3 # order of derivative
#    hs = np.logspace(-3, 8, n_h)
#    abserrs = np.zeros((n_h, n_x))
#    relerrs = np.zeros((n_h, n_x))
#    f = lambda x: np.sqrt(x)
#    df = lambda n, x: (-1)**(n-1)*math.prod([2*i+1 for i in range(n-1)])/2**n*x**(-(2*n-1)/2)
#    for i, h in enumerate(hs):
#        for j, x0 in enumerate(x0s):
#            D, x = riccati.cheb(n)
#            u = h/2*x + x0 + h/2
#            #print("nodes: ", u)
#            df_num = f(u)
#            #print("f(nodes): ", df_num)
#            for k in range(o):
#                df_num = 2/h*np.matmul(D, df_num)
#                #print("f^({})(nodes): ".format(k+1), df_num)
#            df_ana = df(o, u)
#            #print("At h = {}, n = {}, df_num = {}, df_ana = {}".format(h, n, df_num, df_ana))
#            abserrs[i, j] = max(np.abs((df_num - df_ana)))
#            relerrs[i, j] = max(np.abs((df_num - df_ana)/df_ana))
#     
#    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(5, 6))
#    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab20c.colors)
#    ax[0].set_title("Error in numerical differentiation based on Cheb nodes, $f(x) = \sqrt{x}$, $n = 32$, $o = 3$", wrap=True)
#    ax[1].set_xlabel("$h$")
#    ax[0].set_ylabel("Absolute error $|\Delta y|$")
#    ax[1].set_ylabel("Relative error $|\Delta y/y|$")
#    for i, x0 in enumerate(x0s):
#        ax[0].loglog(hs, abserrs[:, i], label='x0 = {}'.format(x0), color='C{}'.format(i))
#        ax[1].loglog(hs, relerrs[:, i], color='C{}'.format(i))
#    ax[0].legend()
#    plt.tight_layout()
#    #plt.savefig("num_diff_airy_vary_x0.pdf")
#    plt.show()

def test_choose_stepsize():
    w = lambda x: np.sqrt(x) 
    n_h = 1000
    x0 = 1e6
    hs = np.logspace(np.log10(x0)-5, np.log10(x0)+3, n_h)
    errs = np.zeros_like(hs)
    p = 32
    q = p//2
    for i, h in enumerate(hs): 
        t = np.random.uniform(low = x0, high = x0+h, size = q)
        s = x0 + h/2 + h/2*riccati.cheb(p-1)[1]
        V = np.ones((p, p))
        R = np.ones((q, p))
        for j in range(1, p):
            V[:, j] = V[:, j-1]*s
            R[:, j] = R[:, j-1]*t
        L = np.linalg.solve(V.T, R.T).T
        wana = w(t)
        west = np.matmul(L, w(s))
        maxwerr = max(np.abs((west - wana)/west))
        errs[i] = maxwerr 

    plt.figure()
    plt.loglog(hs, errs)
    plt.show()

def test_osc_step():
    w = lambda x: np.sqrt(x) #+ 1e-15*np.random.randint(-9, high=9, size=x.size)
    x0 = 1e4
    n_h = 200; 
    eps = 1e-10
    epsh = 1e-14 
    hs = np.logspace(np.log10(x0)-5, np.log10(x0)+3, n_h)
    y0 = mpmath.airyai(-x0)
    dy0 = -mpmath.airyai(-x0, derivative=1)
    y_errs = np.zeros(n_h)
    ress = np.zeros(n_h)
    for i, h in enumerate(hs):
        y_ana = mpmath.airyai(-(x0+h))
        y, dy, err, s = riccati.osc_step(w, x0, h, y0, dy0, epsres = eps)
        y_errs[i] = np.abs((y - y_ana)/y_ana)
        ress[i] = err

    # Check what solver would have chosen as stepsize
    hini = 2*x0
    hest = riccati.choose_stepsize(w, x0, hini, epsh = epsh)

    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(15, 12))
    ax[0].set_title("Error and residual in the Barnett series as a function of stepsize, Airy, $x_0 = ${}, $\epsilon = ${}".format(x0, eps))
    ax[1].set_xlabel("$h$")
    ax[0].set_ylabel("Residual $R[y](x)$")
    ax[1].set_ylabel("Relative error $|\Delta y/y|$")
    ax[0].loglog(hs, ress)
    ax[1].loglog(hs, y_errs)
    ax[1].axvline(x = hini, ls = '--', label = 'initial step estimate', color='C1')
    ax[1].axvline(x = hest, label = 'chosen stepsize', color='C1')
    ax[1].legend()
#    ax[1].set_ylim((1e-16, 1e0))
    plt.tight_layout()
    plt.show()



