import numpy as np
import riccati
import scipy.special as sp
import matplotlib
#matplotlib.use("Qt5Agg")
from matplotlib import pyplot as plt
import math
#import mpmath

def legendre():
    m = int(1e8)
    w = lambda x: np.sqrt(m*(m+1)/(1-x**2))
    g = lambda x: -x/(1-x**2)
    dw = lambda x: np.sqrt(m*(m+1))*x/np.sqrt((1 - x**2)**3) 
    dg = lambda x: - (1 + x**2)/(1 - x**2)**2
    epsf = 0.1 # How far from the singularity we stop integrating
    xi = 0.1
    xf = 1 - epsf
    xcts = np.linspace(xi, xf, 1000)
#    ycts = sp.eval_legendre(m, xcts)

    # Plot frequency and damping
    fig, ax = plt.subplots(2, 1, sharex = True)
    ax[0].set_ylabel('Frequency, $\omega(t)$')
    ax[1].set_ylabel('Damping, $\gamma(t)$')
    ax[1].set_xlabel('Time, $t$')
    ax[0].plot(xcts, w(xcts), label='$\omega$')
    ax[0].plot(xcts, w(xcts)/dw(xcts), label="$\omega/\omega'$")
    ax[1].plot(xcts, g(xcts), label='$\gamma$')
    ax[1].plot(xcts, g(xcts)/dg(xcts), label="$\gamma/\gamma'$")
    ax[0].legend()
    ax[1].legend()
    plt.tight_layout()
    plt.show()

    eps = 1e-10
    epsh = 1e-12
    yi = sp.eval_legendre(m, xi)
    Pmm1_xi = sp.eval_legendre(m-1, xi)
    dyi = m/(xi**2 - 1)*(xi*yi - Pmm1_xi)
    xs, ys, dys, ss, ps, stypes = riccati.solve(w, g, xi, xf, yi, dyi, eps = eps, epsh = epsh)
    xs = np.array(xs)
    ys = np.array(ys)
    ss = np.array(ss)
    ps = np.array(ps)
    stypes = np.array(stypes)
    ytrue = sp.eval_legendre(m, xs)
    yerr = np.abs((ytrue - ys)/ytrue)
    print(stypes)
    fig, ax = plt.subplots(2, 1)
    ax[0].set_ylabel('numerical solution, $u(x)$')
    ax[1].set_ylabel('relative error, $|\Delta y/y|$')
#    ax[0].plot(xcts, ycts, color='black', lw=0.7, label='Analytic solution')
    ax[0].plot(xs[stypes==1], ys[stypes==1], '.', color='C0', label='Riccati step')
    ax[0].plot(xs[stypes==0], ys[stypes==0], '.', color='C1', label='Chebyshev step')
    ax[1].loglog(xs, yerr)
    ax[0].legend()
    plt.tight_layout()
    plt.show()

def assoclegendre():
    m = 1e1
    l = 1e6
    epsf = 0.1
    w = lambda x: np.lib.scimath.sqrt(l*(l+1)/(1-x**2) - m**2/(1-x**2)**2)
    g = lambda x: -x/(1-x**2)
    xi = 0.0
    xf = 1 - epsf
    eps = 1e-6
    epsh = 1e-12
    yis, dyis = sp.lpmn(m, l, xi)
    yi = yis[int(m), -1]
    dyi = dyis[int(m), -1]
    print('yi, dyi', yi, dyi)
    xs, ys, dys, ss, ps, stypes = riccati.solve(w, g, xi, xf, yi, dyi, eps = eps, epsh = epsh)
    xs = np.array(xs)
    ys = np.array(ys)
    ss = np.array(ss)
    ps = np.array(ps)

    ytrue = np.array([sp.lpmn(m, l, x)[0][int(m), -1] for x in xs])
    yerr = np.abs((ytrue - ys)/ytrue)
    hs = np.array([xs[i] - xs[i-1] for i in range(1, xs.shape[0])])
    wiggles = np.array([sum(ps[:i+1])/(2*np.pi) for i in range(ps.shape[0])])
    print('xs', xs)
    print('ys', ys)
    print('ytrue', ytrue)
    print('yerr', yerr)

    fig, ax = plt.subplots(2, 1, sharex = True)
    ax[0].set_ylabel('Stepsize, $h(x)$')
    ax[0].loglog(xs[:-1], hs)
    ax[1].loglog(xs, yerr, color='C0')
    ax[1].loglog(xs[ss==True], yerr[ss==True], '.', color='C0', label='Converged')
    ax[1].loglog(xs[ss==False], yerr[ss==False], '.', color='C1', label='Diverged before $\epsilon$ was hit')
    ax[1].loglog(xs, eps*np.ones_like(xs), color='black')
    ax[1].set_ylabel('Relative error, $|\Delta y/y|$')
    ax[1].set_xlabel('$x$') 
    ax[0].set_title('Numerical associated Legendre function, $m = ${}, $l = ${}, $\epsilon = ${}, $\epsilon_h = ${}'.format(m, l, eps, epsh))
    ax[1].loglog(xs[1:], wiggles*np.finfo(float).eps, color='C2', label='max theoretical accuracy')
    ax[1].legend()
    plt.tight_layout()
    plt.show()


legendre()
