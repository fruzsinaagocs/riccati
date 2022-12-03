import numpy as np
import riccati
import scipy.special as sp
from scipy.integrate import solve_ivp
import matplotlib
from matplotlib import pyplot as plt
import math
import mpmath
import os
from num2tex import num2tex
from pathlib import Path
import time
#import cProfile, pstats
import pyoscode
#from scipy.optimize import root_scalar
from matplotlib.legend_handler import HandlerTuple
import pandas
#import sys
import subprocess

class HandlerTupleVertical(HandlerTuple):
    def __init__(self, **kwargs):
        HandlerTuple.__init__(self, **kwargs)

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        # How many lines are there.
        numlines = len(orig_handle)
        handler_map = legend.get_legend_handler_map()

        # divide the vertical space where the lines will go
        # into equal parts based on the number of lines
        height_y = (height / numlines)

        leglines = []
        for i, handle in enumerate(orig_handle):
            handler = legend.get_legend_handler(handler_map, handle)

            legline = handler.create_artists(legend, handle,
                                             xdescent,
                                             (2*i + 1)*height_y,
                                             width,
                                             2*height,
                                             fontsize, trans)
            leglines.extend(legline)

        return leglines

class counter:
    """
    Wraps a function to keep track of how many times it's been called.
    """

    def __init__(self, f):
        self.f = f
        self.count = 0

    def __call__(self, *args, **kwargs):
        self.count += 1
        return self.f(*args, **kwargs)

def airy():
    w = lambda x: np.sqrt(x)
    g = lambda x: np.zeros_like(x)
    xi = 1e0
    xf = 1e8
    eps = 1e-12
    epsh = 1e-13
    yi = sp.airy(-xi)[0] + 1j*sp.airy(-xi)[2]
    dyi = -sp.airy(-xi)[1] - 1j*sp.airy(-xi)[3]
    info = riccati.setup(w, g, n = 32, p = 32)
    xs, ys, dys, ss, ps, stypes = riccati.solve(info, xi, xf, yi, dyi, eps = eps, epsh = epsh)
    print(xs, ys)
    xs = np.array(xs)
    ys = np.array(ys)
    ss = np.array(ss)
    ps = np.array(ps)
    stypes = np.array(stypes)

    xcts = np.logspace(np.log10(xi), np.log10(65), 5000)
    ycts = np.array([sp.airy(-x)[0] for x in xcts])
    ytrue = np.array([mpmath.airyai(-x) + 1j*mpmath.airybi(-x) for x in xs])
    yerr = np.abs((ytrue - ys)/ytrue)
    hs = np.array([xs[i] - xs[i-1] for i in range(1, xs.shape[0])])
    wiggles = np.array([sum(ps[:i+1]) for i in range(ps.shape[0])])

    tab20c = matplotlib.cm.get_cmap('tab20c')
    blue1 = tab20c.colors[0]
    blue2 = tab20c.colors[1]
    grey2 = tab20c.colors[-3]
    
    plt.style.use('riccatipaper') 
    fig, ax = plt.subplots(2, 2, figsize=(6, 4))
    # Numerical solution
    ax[0,0].semilogx(xcts, ycts, color='black', lw = 0.7, label='analytic solution')
    ax[0,0].semilogx(xs[stypes==0], ys[stypes==0], 'x', color='C0', label='Chebyshev step')
    ax[0,0].semilogx(xs[stypes==1], ys[stypes==1], '.', color='C1', label='Riccati step')
    ax[0,0].set_ylabel('$\Re\left(u(t)\\right)$')
    ax[0,0].set_xlim((1, 65))
    ax[0,0].set_ylim((-0.5, 0.7))
    ax[0,0].legend()
    # Stepsize
    ax[0,1].loglog(xs[:-1], hs, color='black')
    ax[0,1].loglog(np.logspace(0,8,10), np.logspace(0,8,10), '--', color=blue2)
    ax[0,1].set_ylabel('stepsize, $h$')
    ax[0,1].annotate('$\propto t$', (5e3, 1.7e3), rotation=30, c=blue1)
    ax[0,1].set_xlim((1e0, 1e8))
    # Error
    print(xs, yerr)
    ax[1,0].loglog(xs[ss==True], yerr[ss==True], color='black')
    ax[1,0].loglog(xs, eps*np.ones_like(xs), '--', color=grey2, label='$\\varepsilon$')
    ax[1,0].loglog(xs[1:], wiggles*np.finfo(float).eps, '--', color=blue2,\
            label='$\kappa\cdot \\varepsilon_{\mathrm{mach}}$')
    ax[1,0].set_xlabel('$t$')
    ax[1,0].set_ylabel('relative error, $|\Delta u/u|$')
    ax[1,0].set_xlim((xi, xf))
    ax[1,0].legend()
    # Wiggles
    ax[1,1].loglog(xs[1:], ps/(2*np.pi), color='black')
    ax[1,1].set_ylabel('$n_{\mathrm{osc}}$ per step') 
    ax[1,1].set_xlabel('$t$') 
    ax[1,1].loglog(np.logspace(0,8,10), np.logspace(0,8*3/2,10), '--', color=blue2)
    ax[1,1].annotate('$\propto t^{\\frac{3}{2}}$', (5e3, 3e6), rotation=30, c=blue1)
    ax[1,1].set_xlim((1e0, 1e8))
    plt.show()
#    plt.savefig("../plots/airy-numsol-fixncheb-16-epsrel.pdf")

def convergence():
    w = lambda x: np.sqrt(x)
    g = lambda x: np.zeros_like(x)
    xi = 1e0
    xfs = np.array([1e2, 1e3, 1e4, 1e5])
    epss = np.logspace(-12, -4, num = 100)
    errs = np.zeros((xfs.shape[0], epss.shape[0]))
    epsh = 1e-13
    yi = sp.airy(-xi)[0] + 1j*sp.airy(-xi)[2]
    dyi = -sp.airy(-xi)[1] - 1j*sp.airy(-xi)[3]
    wiggles = np.zeros(xfs.shape[0])
    info = riccati.setup(w, g, n = 32, p = 32)
    for i, eps in enumerate(epss):
        for j, xf in enumerate(xfs):
            xs, ys, dys, ss, ps, stypes = riccati.solve(info, xi, xf, yi, dyi, eps = eps, epsh = epsh)
            xs = np.array(xs)
            ys = np.array(ys)
            ytrue = np.array([mpmath.airyai(-x) + 1j*mpmath.airybi(-x) for x in xs])
            yerrs = np.abs((ytrue - ys)/ytrue)
            yerr = np.max(yerrs)
            print(yerr)
            errs[j,i] = yerr
            wiggles[j] = sum(ps)
    
    tab20c = matplotlib.cm.get_cmap('tab20c')
    plt.style.use('riccatipaper') 
    fig, ax1 = plt.subplots(1, 1, figsize=(3, 3))

    cond0 = np.ones_like(epss)*np.finfo(float).eps*wiggles[0]
    lower0 = np.where(cond0 > epss, cond0, None)
    l1, = ax1.loglog(epss, errs[0,:], lw = 1.2, color=tab20c.colors[0*4])
    l2, = ax1.loglog(epss, lower0, ':', lw = 1.0, color=tab20c.colors[0*4 + 1])

    cond1 = np.ones_like(epss)*np.finfo(float).eps*wiggles[1]
    lower1 = np.where(cond1 > epss, cond1, None)
    l3, = ax1.loglog(epss, errs[1,:], ls = (0, (5, 1)), lw = 1.2, color=tab20c.colors[1*4])
    l4, = ax1.loglog(epss, lower1, ':', lw = 1.0, color=tab20c.colors[1*4 + 1])

    cond2 = np.ones_like(epss)*np.finfo(float).eps*wiggles[2]
    lower2 = np.where(cond2 > epss, cond2, None)
    l5, = ax1.loglog(epss, errs[2,:], ls = (0, (5, 5)), lw = 1.2, color=tab20c.colors[2*4])
    l6, = ax1.loglog(epss, lower2, ':', lw = 1.0, color=tab20c.colors[2*4 + 1])

    cond3 = np.ones_like(epss)*np.finfo(float).eps*wiggles[3]
    lower3 = np.where(cond3 > epss, cond3, None)
    l7, = ax1.loglog(epss, errs[3,:], ls = '-.', lw = 1.2, color=tab20c.colors[3*4])
    l8, = ax1.loglog(epss, lower3, ':', lw = 1.0, color=tab20c.colors[3*4 + 1])

    ax1.loglog(epss, epss, '--', color='grey')
    ax1.set_xlabel('tolerance, $\\varepsilon$')
    ax1.set_ylabel('relative error, $|\Delta u/u|$')
    l = ax1.legend([l1, l3, l5, l7, (l2, l4, l6, l8)], ['$t_1 = 10^{}$'.format(int(np.log10(xfs[0]))),'$t_1 = 10^{}$'.format(int(np.log10(xfs[1]))),'$t_1 = 10^{}$'.format(int(np.log10(xfs[2]))),'$t_1 = 10^{}$'.format(int(np.log10(xfs[3]))), '$\kappa \cdot \\varepsilon_{\mathrm{mach}}$'], handler_map = {tuple: HandlerTupleVertical()})
#    plt.show()
    ax1.set_xlim((1e-12,1e-4))
    ax1.set_ylim((1e-12,1.5e-4))
#    plt.savefig("../plots/convergence-fixn.pdf")
    plt.show()

def legendre(m):
    m = int(m)
    global w_count, g_count
    w_count = 0
    g_count = 0
#    w = lambda x: np.sqrt(m*(m+1)/(1-x**2))
#    g = lambda x: -x/(1-x**2)
#    w_counted = counter(w)
#    g_counted = counter(g)

    def w(x):
        global w_count
        try:
            w_count += x.shape[0]
#            print(x.shape)
        except:
            w_count += 1
#            print('1')
        return np.sqrt(m*(m+1)/(1-x**2))

    def g(x):
        global g_count
        try:
            g_count += x.shape[0]
#            print(g.shape)
        except:
            g_count += 1
#            print('1')
        return -x/(1-x**2)

    epsf = 0.1 # How far from the singularity we stop integrating
    xi = 0.1
    xf = 1 - epsf
    xcts = np.linspace(xi, xf, 1000)
#    ycts = sp.eval_legendre(m, xcts)

    eps = 1e-12 #, n = p = 16
    epsh = 1e-13
    yi = sp.eval_legendre(m, xi)
    Pmm1_xi = sp.eval_legendre(m-1, xi)
    dyi = m/(xi**2 - 1)*(xi*yi - Pmm1_xi)
    N = 1

    # Time this process
    start = time.time_ns()
#    profiler = cProfile.Profile()
#    profiler.enable()
    for i in range(N):
        xs, ys, dys, ss, ps, stypes, statdict = riccati.solve(w, g, xi, xf, yi, dyi, eps = eps, epsh = epsh, n = 16, p = 16)
#    profiler.disable()
    end = time.time_ns()

#    stats = pstats.Stats(profiler).sort_stats('tottime')
#    stats.print_stats()
#    stats.dump_stats('output-numba.pstats')

    xs = np.array(xs)
    ys = np.array(ys)
    ss = np.array(ss)
    ps = np.array(ps)
    stypes = np.array(stypes)
    print(statdict)
    ytrue = sp.eval_legendre(m, xs)
    yerr = np.abs((ytrue - ys)/ytrue)
#    fig, ax = plt.subplots(2, 1)
#    ax[0].set_ylabel('numerical solution, $u(x)$')
#    ax[1].set_ylabel('relative error, $|\Delta y/y|$')
##    ax[0].plot(xcts, ycts, color='black', lw=0.7, label='Analytic solution')
#    ax[0].plot(xs[stypes==1], ys[stypes==1], '.', color='C0', label='Riccati step')
#    ax[0].plot(xs[stypes==0], ys[stypes==0], '.', color='C1', label='Chebyshev step')
#    ax[1].loglog(xs, yerr)
#    ax[0].legend()
#    plt.tight_layout()
##    plt.show()

    # Write LaTeX table to file
    round_to_n = lambda n, x: x if x == 0 else round(x, - int(math.floor(math.log10(abs(x)))) + (n-1))
    steps = (statdict["cheb steps"][0] + statdict["ricc steps"][0], statdict["cheb steps"][1] + statdict["ricc steps"][1])
    steps_ricc = statdict['ricc steps']
    steps_cheb = statdict['cheb steps']
    n_fevals = (w_count + g_count)/N
    n_LS = statdict["linear solves"]
    n_LU = statdict["LU decomp"]
    n_sub = statdict["substitution"]
    outputf = "riccati-paper/tables/legendre-fastest-morestuff.tex"
    outputpath = Path(outputf)
    outputpath.touch(exist_ok = True)
    print("time/s: ", (end - start)*1e-9)
    runtime = (end - start)*1e-9/N
    evaltime = 0.0 #TODO
    lines = ""
    if os.stat(outputf).st_size != 0:
        with open(outputf, 'r') as f:
            lines = f.readlines()
            lines = lines[:-2] # Remove trailing \hline-s and \end{tabular}
    with open(outputf, 'w') as f:
        if lines == "": 
            f.write("\\begin{tabular}{l c c c c c c c c c}\n")
            f.write("\hline \hline \n$n$  &  $\max|\Delta u/u|$  &  $t_{\mathrm{solve}}$/\si{\s} &  $n_{\mathrm{s,slo}}$  &  $n_{\mathrm{s,osc}}$  &  $n_{\mathrm{s,tot}}$  &  $n_{\mathrm{f}}$  &  $n_{\mathrm{LS}}$  &  $n_{\mathrm{LU}}$  &  $n_{\mathrm{sub}}$ \\\\ \hline\n")
        for line in lines:
            f.write(line)
        f.write("$10^{}$  &  ${}$  &  ${}$  &  ${}$  &  ${}$  &  ${}$  &  ${}$  &  ${}$  &  ${}$  &  ${}$  &  ${}$\\\\ \n".format(int(np.log10(m)), num2tex(round_to_n(3, max(yerr))), num2tex(round_to_n(3, runtime)), evaltime, steps_cheb, steps_ricc, steps, round(n_fevals), n_LS, n_LU, n_sub))
        f.write("\hline \hline\n\end{tabular}\n")

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
    xs, ys, dys, ss, ps, stypes, statdict = riccati.solve(w, g, xi, xf, yi, dyi, eps = eps, epsh = epsh)
    xs = np.array(xs)
    ys = np.array(ys)
    ss = np.array(ss)
    ps = np.array(ps)

    ytrue = np.array([sp.lpmn(m, l, x)[0][int(m), -1] for x in xs])
    yerr = np.abs((ytrue - ys)/ytrue)
    hs = np.array([xs[i] - xs[i-1] for i in range(1, xs.shape[0])])
    wiggles = np.array([sum(ps[:i+1]) for i in range(ps.shape[0])])
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
    ax[1].set_xlabel('$t$') 
    ax[0].set_title('Numerical associated Legendre function, $m = 10^{}$, $l = ${}, $\epsilon = ${}, $\epsilon_h = ${}'.format(int(np.log10(m)), l, eps, epsh))
    ax[1].loglog(xs[1:], wiggles*np.finfo(float).eps, color='C2', label='max theoretical accuracy')
    ax[1].legend()
    plt.tight_layout()
    plt.show()

def residual():
    w = lambda x: np.sqrt(x)
    g = lambda x: np.zeros_like(x)
    x0 = 1.8
    h = 14.0
    eps = 1e-12
    ks = [0, 1, 3, 9]
    y0 = sp.airy(-x0)[0] + 1j*sp.airy(-x0)[2]
    dy0 = -sp.airy(-x0)[1] - 1j*sp.airy(-x0)[3]
    xcts = np.linspace(x0, x0+h, 1200)
    ycts = np.array([sp.airy(-x)[0] + 1j*sp.airy(-x)[2] for x in xcts])

    tab20c = matplotlib.cm.get_cmap('tab20c')
    blue1 = tab20c.colors[0]
    blue2 = tab20c.colors[1]
    grey2 = tab20c.colors[-3]
    orange1 = tab20c.colors[4]

    plt.style.use('riccatipaper') 

    N = 10 # max number of iters
    ks = np.linspace(0, N, N+1, dtype=int)
    ms = [int(1e1), int(1e2), int(1e3), int(1e4)]
    g = lambda x: np.zeros_like(x)
    x0 = 0
    h = 0.5
    n = 16
    p = 16
    eps = 1e-12
    epsh = 1e-13
    errs = np.zeros(N+1)
    info = riccati.setup(w, g, n = n, p = p)

    fig, ax1 = plt.subplots(1, 1, figsize = (3, 3))
    ax1.set_xlabel("iteration number, $j$")
    ax1.set_ylabel("maximum residual, $\max\limits_{t \in [0, " + "{}".format(h) + "]}R\left[x_j\\right]$")

    m = ms[0]
    w = lambda x: np.sqrt(m**2 - 1)/(1 + x**2)
    for i, k in enumerate(ks):
        bursty = lambda x: np.sqrt(1 + x**2)/m*(np.cos(m*np.arctan(x)) + 1j*np.sin(m*np.arctan(x))) 
        burstdy = lambda x: 1/np.sqrt(1 + x**2)/m*((x + 1j*m)*np.cos(m*np.arctan(x))\
               + (-m + 1j*x)*np.sin(m*np.arctan(x)))
        y0 = bursty(x0)
        dy0 = burstdy(x0)   
        xscaled = x0 + h/2 + h/2*info.xn
        info.wn = w(xscaled)
        info.gn = g(xscaled)
        err = riccati.osc_step(info, x0, h, y0, dy0, epsres = eps, plotting = True, k = k)
        errs[i] = err

    l1, = ax1.semilogy(ks, errs, '.-', color=tab20c.colors[0*4])
    l2, = ax1.semilogy(ks, 10.0**(np.log10(m)*(1.0-ks)), '--', color=tab20c.colors[0*4+1])
    
    m = ms[1]
    w = lambda x: np.sqrt(m**2 - 1)/(1 + x**2)
    for i, k in enumerate(ks):
        bursty = lambda x: np.sqrt(1 + x**2)/m*(np.cos(m*np.arctan(x)) + 1j*np.sin(m*np.arctan(x))) 
        burstdy = lambda x: 1/np.sqrt(1 + x**2)/m*((x + 1j*m)*np.cos(m*np.arctan(x))\
               + (-m + 1j*x)*np.sin(m*np.arctan(x)))
        y0 = bursty(x0)
        dy0 = burstdy(x0)   
        xscaled = x0 + h/2 + h/2*info.xn
        info.wn = w(xscaled)
        info.gn = g(xscaled)
        err = riccati.osc_step(info, x0, h, y0, dy0, epsres = eps, plotting = True, k = k)
        errs[i] = err

    l3, = ax1.semilogy(ks, errs, 'o-', color=tab20c.colors[1*4])
    l4, = ax1.semilogy(ks, 10.0**(np.log10(m)*(1.0-ks)), '--', color=tab20c.colors[1*4+1])
 
    m = ms[2]
    w = lambda x: np.sqrt(m**2 - 1)/(1 + x**2)
    for i, k in enumerate(ks):
        bursty = lambda x: np.sqrt(1 + x**2)/m*(np.cos(m*np.arctan(x)) + 1j*np.sin(m*np.arctan(x))) 
        burstdy = lambda x: 1/np.sqrt(1 + x**2)/m*((x + 1j*m)*np.cos(m*np.arctan(x))\
               + (-m + 1j*x)*np.sin(m*np.arctan(x)))
        y0 = bursty(x0)
        dy0 = burstdy(x0)   
        xscaled = x0 + h/2 + h/2*info.xn
        info.wn = w(xscaled)
        info.gn = g(xscaled)
        err = riccati.osc_step(info, x0, h, y0, dy0, epsres = eps, plotting = True, k = k)
        errs[i] = err

    l5, = ax1.semilogy(ks, errs, '^-', color=tab20c.colors[2*4])
    l6, = ax1.semilogy(ks, 10.0**(np.log10(m)*(1.0-ks)), '--', color=tab20c.colors[2*4+1])
 
    m = ms[3]
    w = lambda x: np.sqrt(m**2 - 1)/(1 + x**2)
    for i, k in enumerate(ks):
        bursty = lambda x: np.sqrt(1 + x**2)/m*(np.cos(m*np.arctan(x)) + 1j*np.sin(m*np.arctan(x))) 
        burstdy = lambda x: 1/np.sqrt(1 + x**2)/m*((x + 1j*m)*np.cos(m*np.arctan(x))\
               + (-m + 1j*x)*np.sin(m*np.arctan(x)))
        y0 = bursty(x0)
        dy0 = burstdy(x0)   
        xscaled = x0 + h/2 + h/2*info.xn
        info.wn = w(xscaled)
        info.gn = g(xscaled)
        err = riccati.osc_step(info, x0, h, y0, dy0, epsres = eps, plotting = True, k = k)
        errs[i] = err

    l7, = ax1.semilogy(ks, errs, 'x-', color=tab20c.colors[3*4])
    l8, = ax1.semilogy(ks, 10.0**(np.log10(m)*(1.0-ks)), '--', color=tab20c.colors[3*4+1])
 
#, label='$\omega(t_i) = 10^{}$'.format(int(np.log10(m)))
    ax1.set_xlim((0,N))
    ax1.set_ylim((1e-16, 1e6))
    l = ax1.legend([l1, l3, l5, l7, (l2, l4, l6, l8)], [
        '$\omega_{\mathrm{max}} = $'+'$\sqrt{' + '10^{}'.format(2*int(np.log10(ms[0]))) + '- 1}$',
        '$\omega_{\mathrm{max}} = $'+'$\sqrt{' + '10^{}'.format(2*int(np.log10(ms[1]))) + '- 1}$',
        '$\omega_{\mathrm{max}} = $'+'$\sqrt{' + '10^{}'.format(2*int(np.log10(ms[2]))) + '- 1}$',
        '$\omega_{\mathrm{max}} = $'+'$\sqrt{' + '10^{}'.format(2*int(np.log10(ms[3]))) + '- 1}$',
        '$\propto \omega_{\mathrm{max}}^{-j}$'], 
        handler_map = {tuple: HandlerTupleVertical()})
#    plt.savefig("../plots/residual-k-n8-h025.pdf")
    plt.show()

def Bremer237(l, n, eps, epsh, outdir, rdc = True, wkbmarching = False,\
              kummer = False, oscode = False, rk = False):
    """
    Solves problem (237) from Bremer's "On the numerical solution of second
    order ordinary differential equations in the high-frequency regime" paper.
    """
    global w_count, g_count
    w_count = 0
    g_count = 0

    def w(x):
        global w_count
        try:
            w_count += x.shape[0]
        except:
            w_count += 1
        return l*np.sqrt(1 - x**2*np.cos(3*x))

    def g(x):
        global g_count
        try:
            g_count += x.shape[0]
        except:
            g_count += 1
        return np.zeros_like(x)

    # For the reference solution 
    def f(t, y):
        yp = np.zeros_like(y)
        yp[0] = y[1]
        yp[1] = -l**2*(1 - t**2*np.cos(3*t))*y[0]
        return yp

    xi = -1.0
    xf = 1.0

    eps = eps
    yi = 0.0
    dyi = l
    yi_vec = np.array([yi, dyi])

    # Utility function for rounding to n significant digits
    round_to_n = lambda n, x: x if x == 0 else round(x, - int(math.floor(math.log10(abs(x)))) + (n-1))
        
    # Reference solution and its reported error
    reftable = "/mnt/home/fagocs/riccati-paper/tables/new/ref/eq237.txt"
    refarray = np.genfromtxt(reftable, delimiter=',')
    ls = refarray[:,0]
    ytrue = refarray[abs(ls -l) < 1e-8, 1]
    errref = refarray[abs(ls -l) < 1e-8, 2]

    if rdc:
        N = 1000 # Number of repetitions for timing
        epsh = epsh
        n = n
        p = n

        start = time.time_ns()
        info = riccati.setup(w, g, n = n, p = p)
        for i in range(N):
            xs, ys, dys, ss, ps, stypes = riccati.solve(info, xi, xf, yi, dyi, eps = eps, epsh = epsh, hard_stop = True)
        end = time.time_ns()
        xs = np.array(xs)
        ys = np.array(ys)
        ss = np.array(ss)
        ps = np.array(ps)
        stypes = np.array(stypes)
        # Counted some statistics N times, correct for that:
        info.n_chebstep /= N
        info.n_riccstep /= N
        info.n_LU = (info.n_LU - 1)/N + 1
        info.n_sub /= N
        info.n_LS = (info.n_LS - 1)/N + 1
    
        statdict = info.output(stypes)

        # Compute statistics
        steps = (statdict["cheb steps"][0] + statdict["ricc steps"][0], statdict["cheb steps"][1] + statdict["ricc steps"][1])
        steps_ricc = statdict['ricc steps']
        steps_cheb = statdict['cheb steps']
        n_fevals = (w_count + g_count)/N
        n_LS = statdict["linear solves"]
        n_LU = statdict["LU decomp"]
        n_sub = statdict["substitution"]
        runtime = (end - start)*1e-9/N
        yerr = np.abs((ytrue - ys[-1])/ytrue)

      
        # Write to txt file
        # Create dir
        os.system("mkdir -p {}/rdc".format(outdir))
        outputf = outdir + "/rdc/bremer237-rdc.txt" 
        outputpath = Path(outputf)
        outputpath.touch(exist_ok = True)
        lines = ""
        if os.stat(outputf).st_size != 0:
            with open(outputf, 'r') as f:
                lines = f.readlines()
        with open(outputf, 'w') as f:
            if lines == "": 
                f.write("# method, l, eps, relerr, tsolve, n_s_osc_att, n_s_osc_suc, n_s_slo_att, n_s_slo_suc, n_s_tot_att, n_s_tot_suc, n_f, n_LS, n_LU, n_sub, errlessref, params\n")
            for line in lines:
                f.write(line)
            f.write("{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}".format("rdc",\
                    l, eps, round_to_n(3, max(yerr)), round_to_n(3, runtime), steps_ricc[0], steps_ricc[1],\
                    steps_cheb[0], steps_cheb[1], steps[0], steps[1], round(n_fevals), n_LS, n_LU, n_sub,\
                    (yerr < errref)[0], "(n = {}; p = {}; epsh = {})".format(n, p, epsh)))
            f.write("\n")

    if oscode:
        w_count = 0
        g_count = 0
    
        if eps < 1e-8 and l < 1e4:
            N = 10
        else:
            N = 100
    
        # Time this process
        start = time.time_ns()
        for i in range(N):
            solution = pyoscode.solve_fn(w, g, xi, xf, yi, dyi, rtol = eps)
        end = time.time_ns()
        xs = solution['t']
        ys = solution['sol']
        ss = solution['types']
    
        xs = np.array(xs)
        ys = np.array(ys)
        ss = np.array(ss)
      
        yerr = np.abs((ytrue - ys[-1])/ytrue)
    
        # Write LaTeX table to file
        steps = ss.shape[0] - 1
        steps_wkb = np.count_nonzero(ss)
        steps_rk = steps - steps_wkb 
        n_fevals = (w_count + g_count)/N
        runtime = (end - start)*1e-9/N

        # Write to txt file
        # Create dir
        os.system("mkdir -p {}/oscode".format(outdir))
        outputf = outdir + "/oscode/bremer237-oscode.txt" 
        outputpath = Path(outputf)
        outputpath.touch(exist_ok = True)
        lines = ""
        if os.stat(outputf).st_size != 0:
            with open(outputf, 'r') as f:
                lines = f.readlines()
        with open(outputf, 'w') as f:
            if lines == "": 
                f.write("# method, l, eps, relerr, tsolve, n_s_osc_att, n_s_osc_suc, n_s_slo_att, n_s_slo_suc, n_s_tot_att, n_s_tot_suc, n_f, n_LS, n_LU, n_sub, errlessref, params\n")
            for line in lines:
                f.write(line)
            f.write("{}, {}, {}, {}, {}, , {}, , {}, , {}, {}, , , , {}, {}".format("oscode",\
                    l, eps, round_to_n(3, max(yerr)), round_to_n(3, runtime), steps_wkb, steps_rk,\
                    steps, round(n_fevals), (yerr < errref)[0], "(nrk = default; nwkb = default)"))
            f.write("\n")
      

    if kummer:
        kummerscript = "test_eq237_2"
        # Compile and run fortran code with command-line arguments l and eps
        #currentdir = os.getcwd()
        os.chdir("/mnt/home/fagocs/riccati-paper/code/bremerphasefun")
        subprocess.run(["make", "clean"])
        subprocess.run(["make", kummerscript])
        kummeroutput = subprocess.run(["./{}".format(kummerscript), str(l), str(eps)], capture_output = True)
        kummerstdout = kummeroutput.stdout.decode('utf-8')
        y, runtime, n_fevals = [float(i) for i in kummerstdout.split()]
        yerr = np.abs((ytrue - y)/ytrue)
        print(yerr, ytrue, y)

        # Write to txt file
        # Create dir
        os.system("mkdir -p {}/kummer".format(outdir))
        outputf = outdir + "/kummer/bremer237-kummer.txt" 
        outputpath = Path(outputf)
        outputpath.touch(exist_ok = True)
        lines = ""
        if os.stat(outputf).st_size != 0:
            with open(outputf, 'r') as f:
                lines = f.readlines()
        with open(outputf, 'w') as f:
            if lines == "": 
                f.write("# method, l, eps, relerr, tsolve, n_s_osc_att, n_s_osc_suc, n_s_slo_att, n_s_slo_suc, n_s_tot_att, n_s_tot_suc, n_f, n_LS, n_LU, n_sub, errlessref, params\n")
            for line in lines:
                f.write(line)
            f.write("{}, {}, {}, {}, {}, , , , , , , {}, , , , {}, {}".format("kummer",\
                    l, eps, round_to_n(3, max(yerr)), round_to_n(3, runtime),\
                    round(n_fevals), (yerr < errref)[0], "()"))
            f.write("\n")
 
    
    if wkbmarching:
        if eps < 1e-8 and l < 1e2:
            N = 1000
        elif eps < 1e-8 and l < 1e4:
            N = 100
        elif l < 1e2:
            N = 100
        else:
            N = 100000
        print("N:", N)
        # Write to txt file
        # Create dir
        matlabscript = "bremer237"
        os.system("mkdir -p {}/wkbmarching".format(outdir))
        outputf = "\\\"" +  outdir + "/wkbmarching/bremer237-wkbmarching.txt\\\""
        outputpath = Path(outputf)
        os.chdir("/mnt/home/fagocs/riccati-paper/code/arnoldwkb")
        # Run matlab script (will write file)
        os.system("matlab -batch \"global aeval; {}({}, {}, {}, {}); exit\" ".format(matlabscript, l, eps, N, outputf))

    if rk:
        # We're only running this once because it's slow
        atol = 1e-14
        method = "DOP853"
        f = lambda t, y: np.array([y[1], -l**2*(1 - t**2*np.cos(3*t))*y[0]])
        time0 = time.time_ns()
        sol = solve_ivp(f, [-1, 1], [0, l], method = method, rtol = eps, atol = atol)
        time1 = time.time_ns()
        runtime = (time1 - time0)*1e-9
        nfevalrk = sol.nfev

        err = np.abs((sol.y[0,-1]- ytrue)/ytrue)[0]
        print(sol.status)
        print(sol.message)
        print(sol.success)
   
        # Write to txt file
        # Create dir
        os.system("mkdir -p {}/rk".format(outdir))
        outputf = outdir + "/rk/bremer237-rk.txt" 
        outputpath = Path(outputf)
        outputpath.touch(exist_ok = True)
        lines = ""
        if os.stat(outputf).st_size != 0:
            with open(outputf, 'r') as f:
                lines = f.readlines()
        with open(outputf, 'w') as f:
            if lines == "": 
                f.write("# method, l, eps, relerr, tsolve, n_s_osc_att, n_s_osc_suc, n_s_slo_att, n_s_slo_suc, n_s_tot_att, n_s_tot_suc, n_f, n_LS, n_LU, n_sub, errlessref, params\n")
            for line in lines:
                f.write(line)
            f.write("{}, {}, {}, {}, {}, , , , , , , {}, , , , {}, {}".format("rk",\
                    l, eps, round_to_n(3, err), round_to_n(3, runtime), round(nfevalrk), 
                    (err < errref)[0], "(atol = {}; method = {})".format(atol, method)))
            f.write("\n")
        

def bremer237_timing_fig(outdir):
    # Helper function
    round_to_n = lambda n, x: x if x == 0 else round(x, - int(math.floor(math.log10(abs(x)))) + (n-1))

    # Read in little tables and combine into one pandas dataframe
    outputfs = [outdir + "/{0}/bremer237-{0}.txt".format(method) for method in ["rk", "rdc", "kummer", "oscode", "wkbmarching"]] 
    dfs = []
    for outputf in outputfs:
        df = pandas.read_csv(outputf, sep = ', ')#, index_col = None)
        dfs.append(df)
    data = pandas.concat(dfs, axis = 0)#, ignore_index = True)
    print(data)
    print(data.columns)

    solvernames = data['# method']
    epss = data['eps']
    oscodes = data.loc[solvernames == 'oscode']
    rks = data.loc[solvernames == 'rk']
    wkbs = data.loc[solvernames == 'wkbmarching']
    rdcs = data.loc[solvernames == 'rdc']
    kummers = data.loc[solvernames == 'kummer']
    allosc = oscodes.loc[oscodes['eps'] == 1e-12]
    allrk = rks.loc[rks['eps'] == 1e-12]
    allricc = rdcs.loc[rdcs['eps'] == 1e-12]
    allarn = wkbs.loc[wkbs['eps'] == 1e-12]
    allkummer = kummers.loc[kummers['eps'] == 1e-12]
    losc = allosc['l']
    lrk = allrk['l'] 
    lricc = allricc['l'] 
    larn = allarn['l'] 
    lkum = allkummer['l']
    tosc = allosc['tsolve']
    trk = allrk['tsolve']
    tricc = allricc['tsolve']
    tkum = allkummer['tsolve']
    tarn = allarn['tsolve']
    eosc = allosc['relerr']
    erk = allrk['relerr']
    ericc = allricc['relerr']
    earn = allarn['relerr']
    ekum = allkummer['relerr']

    allosc2 = oscodes.loc[oscodes['eps'] == 1e-6]
    allrk2 = rks.loc[rks['eps'] == 1e-6] 
    allricc2 = rdcs.loc[rdcs['eps'] == 1e-6]
    allarn2 = wkbs.loc[wkbs['eps'] == 1e-6]
    allkummer2 = kummers.loc[kummers['eps'] == 1e-6]

    losc2 = allosc2['l']
    lrk2 = allrk2['l'] 
    lricc2 = allricc2['l'] 
    larn2 = allarn2['l'] 
    lkum2 = allkummer2['l']
    tosc2 = allosc2['tsolve']
    trk2 = allrk2['tsolve']
    tricc2 = allricc2['tsolve']
    tarn2 = allarn2['tsolve']
    tkum2 = allkummer2['tsolve']
    eosc2 = allosc2['relerr']
    erk2 = allrk2['relerr']
    ericc2 = allricc2['relerr']
    earn2 = allarn2['relerr']
    ekum2 = allkummer2['relerr']

    # Bremer 'exclusion zone'
    ebrem = np.array([7e-14, 5e-13, 3e-12, 5e-11, 3e-10, 5e-9, 4e-8])

    # Colourmap
    tab20c = matplotlib.cm.get_cmap('tab20c').colors
    tab20b = matplotlib.cm.get_cmap('tab20b').colors


    plt.style.use('riccatipaper')
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize = (6, 3))
    l1, = ax0.loglog(lrk, trk, '.-', c = tab20c[0*4 + 0])
    l2, = ax0.loglog(losc, tosc, 'o-', c = tab20c[1*4 + 0])
    l3, = ax0.loglog(larn, tarn, '^-', c = tab20c[2*4 + 0])
    l4, = ax0.loglog(lkum, tkum, 'x-', c = tab20c[3*4 + 0])
    l5, = ax0.loglog(lricc, tricc, 'v-', c = tab20b[2*4 + 0])

    l6, = ax0.loglog(lrk2, trk2, marker = '.', ls = '--',  c = tab20c[0*4 + 1])
    l7, = ax0.loglog(losc2, tosc2, marker = 'o', ls = '--', c = tab20c[1*4 + 1])
    l8, = ax0.loglog(larn2, tarn2, marker = '^', ls = '--', c = tab20c[2*4 + 1])
    l9, = ax0.loglog(lkum2, tkum2, marker = 'x', ls = '--', c = tab20c[3*4 + 1])
    l10, = ax0.loglog(lricc2, tricc2, marker = 'v', ls = '--', c = tab20b[2*4 + 1])

    # Invisible lines
    l11, = ax0.loglog(lricc, tricc*1e-5, c = tab20c[0*4 + 0])
    l12, = ax0.loglog(lricc, tricc*1e-5, c = tab20c[1*4 + 0])
    l13, = ax0.loglog(lricc, tricc*1e-5, c = tab20c[2*4 + 0])
    l14, = ax0.loglog(lricc, tricc*1e-5, c = tab20c[3*4 + 0])
    l15, = ax0.loglog(lricc, tricc*1e-5, c = tab20b[2*4 + 0])
    l16, = ax0.loglog(lricc, tricc*1e-5, '--', c = tab20c[0*4 + 1])
    l17, = ax0.loglog(lricc, tricc*1e-5, '--', c = tab20c[1*4 + 1])
    l18, = ax0.loglog(lricc, tricc*1e-5, '--', c = tab20c[2*4 + 1])
    l19, = ax0.loglog(lricc, tricc*1e-5, '--', c = tab20c[3*4 + 1])
    l20, = ax0.loglog(lricc, tricc*1e-5, '--', c = tab20b[2*4 + 1])

    l = ax0.legend([(l1, l6), (l2, l7), (l3, l8), (l4, l9), (l5, l10), (l11, l12, l13, l14, l15), (l16, l17, l18, l19, l20)], ['RK78', '\\texttt{oscode}', 'WKB marching', "Kummer's phase function", 'ARDC', '$\\varepsilon = 10^{-12}$', '$\\varepsilon = 10^{-6}$'], handler_map = {tuple: HandlerTupleVertical()})

    ax1.fill_between(losc, np.ones_like(losc)*1e-14, ebrem, color = 'grey', alpha = 0.4, linewidth=0.0)
    ax1.loglog(losc, np.ones_like(losc)*1e-6, ls = 'dotted', c = 'grey')
    ax1.loglog(losc, np.ones_like(losc)*1e-12, ls = 'dotted', c = 'black')
    l1, = ax1.loglog(lrk, erk, '.-', c = tab20c[0*4 + 0])
    l2, = ax1.loglog(losc, eosc, 'o-', c = tab20c[1*4 + 0])
    l3, = ax1.loglog(larn, earn, '^-', c = tab20c[2*4 + 0])
    l4, = ax1.loglog(lkum, ekum, 'x-', c = tab20c[3*4 + 0])
    l5, = ax1.loglog(lricc, ericc, 'v-', c = tab20b[2*4 + 0])

    l6, = ax1.loglog(lrk2, erk2, marker = '.', ls = '--',  c = tab20c[0*4 + 1])
    l7, = ax1.loglog(losc2, eosc2, marker = 'o', ls = '--', c = tab20c[1*4 + 1])
    l8, = ax1.loglog(larn2, earn2, marker = '^', ls = '--', c = tab20c[2*4 + 1])
    l9, = ax1.loglog(lkum2, ekum2, marker = 'x', ls = '--', c = tab20c[3*4 + 1])
    l10, = ax1.loglog(lricc2, ericc2, marker = 'v', ls = '--', c = tab20b[2*4 + 1])
    # Invisible lines
    
    l11, = ax0.loglog(lricc, tricc*1e-5, c = tab20c[0*4 + 0])
    l12, = ax0.loglog(lricc, tricc*1e-5, c = tab20c[1*4 + 0])
    l13, = ax0.loglog(lricc, tricc*1e-5, c = tab20c[2*4 + 0])
    l14, = ax0.loglog(lricc, tricc*1e-5, c = tab20c[3*4 + 0])
    l15, = ax0.loglog(lricc, tricc*1e-5, c = tab20b[2*4 + 0])
    l16, = ax0.loglog(lricc, tricc*1e-5, '--', c = tab20c[0*4 + 1])
    l17, = ax0.loglog(lricc, tricc*1e-5, '--', c = tab20c[1*4 + 1])
    l18, = ax0.loglog(lricc, tricc*1e-5, '--', c = tab20c[2*4 + 1])
    l19, = ax0.loglog(lricc, tricc*1e-5, '--', c = tab20c[3*4 + 1])
    l20, = ax0.loglog(lricc, tricc*1e-5, '--', c = tab20b[2*4 + 1])

    l = ax1.legend([(l1, l6), (l2, l7), (l3, l8), (l4, l9), (l5, l10), (l11, l12, l13, l14, l15), (l16, l17, l18, l19, l20)], ['RK78', '\\texttt{oscode}', 'WKB marching', "Kummer's phase function", 'ARDC', '$\\varepsilon = 10^{-12}$', '$\\varepsilon = 10^{-6}$'], handler_map = {tuple: HandlerTupleVertical()}, loc = 'upper left')

    ax1.set_ylabel('relative error, $|\Delta u/u|$')
    ax0.set_ylabel('runtime/s, $t_{\mathrm{solve}}$')
    ax1.set_xlabel('$\lambda$')
    ax0.set_xlim((1e1, 1e7))
    ax0.set_ylim((2e-4, 2e5))
    ax1.set_xlim((1e1, 1e7))
    ax1.set_ylim((7e-14, 2e2))
    ax0.set_xlabel('$\lambda$')

#    plt.show()
    plt.savefig('/mnt/home/fagocs/riccati-paper/plots/bremer237-timing-1thread.pdf')


    # Now make LaTeX table from the eps = 1e-12 runs
    outputf = outdir + "/all/bremer237-all-1e-12.tex"
    outputpath = Path(outputf)
    outputpath.touch(exist_ok = True)
    mnames = ['ARDC', "Kummer's phase function", "WKB marching", "\\texttt{oscode}"]
    snames = ["rdc", "kummer", "wkbmarching", "oscode"]
    with open(outputf, 'w') as f:

        # Header
        f.write("\\begin{tabular}{l c l c c c c c c}\n")
        f.write("\hline \hline \n method &  $\lambda$  &  $\max|\Delta u/u|$  &  $t_{\mathrm{solve}}$/\si{\s} &  $n_{\mathrm{s,osc}}$  &  $n_{\mathrm{s,slo}}$  &  $n_{\mathrm{s,tot}}$  &  $n_{\mathrm{f}}$  &  $n_{\mathrm{LS}}$  \\\\ \hline\n")
        
        for mname, sname in zip(mnames, snames):
            alldata = data.loc[solvernames == sname]
            dataeps = alldata.loc[alldata['eps'] == 1e-12] 
            logls = [int(logl) for logl in np.log10(dataeps['l'])]
            relerrs = [num2tex(round_to_n(3, relerr)) if errlessref == False else "\leq {}".format(num2tex(uppererr)) for relerr, errlessref, uppererr in zip(dataeps['relerr'], dataeps['errlessref'], ebrem)]
            tsolves = [num2tex("{:.2e}".format(tsolve)) for tsolve in dataeps['tsolve']]
            if sname == "rdc":
                nosc = ["({}, {})".format(int(nosca), int(noscs)) for noscs, nosca in zip(dataeps['n_s_osc_suc'], dataeps['n_s_osc_att'])]
                nslo = ["({}, {})".format(int(nsloa), int(nslos)) for nslos, nsloa in zip(dataeps['n_s_slo_suc'], dataeps['n_s_slo_att'])]
                nsteps = ["({}, {})".format(int(ntota), int(ntots)) for ntots, ntota in zip(dataeps['n_s_tot_suc'], dataeps['n_s_tot_att'])]
                nls = [int(nl) for nl in dataeps['n_LS']] 
            elif sname != "kummer":
                nosc = [int(nos) for nos in dataeps['n_s_osc_suc']]
                nslo = [int(nsl) for nsl in dataeps['n_s_slo_suc']]
                nsteps = [int(nst) for nst in dataeps['n_s_tot_suc']]
                nls = ["" for nl in dataeps['n_LS']] 
            else:
                nosc = ["" for nos in dataeps['n_s_osc_suc']]
                nslo = ["" for nsl in dataeps['n_s_slo_suc']]
                nsteps = ["" for nst in dataeps['n_s_tot_suc']]
                nls = ["" for nl in dataeps['n_LS']]
            nfeval = [int(fev) for fev in dataeps['n_f']]
            for i, col1, col2, col3, col4, col5, col6, col7, col8 in zip(range(len(logls)), logls, relerrs, tsolves, nosc, nslo, nsteps, nfeval, nls):
                if i == 0:
                    # Only first row of each method has method name
                    f.write("{} & $10^{}$  &  ${}$  &  ${}$  &  ${}$  &  ${}$  &  ${}$  &  ${}$  &  ${}$ \\\\ \n".format(mname, col1, col2, col3, col4, col5, col6, col7, col8))
                else:
                     f.write(" & $10^{}$  &  ${}$  &  ${}$  &  ${}$  &  ${}$  &  ${}$  &  ${}$  &  ${}$\\\\ \n".format(col1, col2, col3, col4, col5, col6, col7, col8))
            # Separator between methods
            f.write("\hline \hline\n")
        # Footer
        f.write("\end{tabular}\n")


def oscode237(l):

    global w_count, g_count
    w_count = 0
    g_count = 0

    def w(x):
        global w_count
        try:
            w_count += x.shape[0]
        except:
            w_count += 1
        return l*np.sqrt(1 - x**2*np.cos(3*x))

    def g(x):
        global g_count
        try:
            g_count += x.shape[0]
        except:
            g_count += 1
        return np.zeros_like(x)

    # For the reference solution 
    def f(t, y):
        yp = np.zeros_like(y)
        yp[0] = y[1]
        yp[1] = -l**2*(1 - t**2*np.cos(3*t))*y[0]
        return yp

    xi = -1.0
    xf = 1.0

    yi = 0.0
    dyi = l
    yi_vec = np.array([yi, dyi])
    N = 100

    # Time this process
    start = time.time_ns()
    for i in range(N):
        solution = oscode.solve_fn(w, g, xi, xf, yi, dyi, rtol = 1e-12)
    end = time.time_ns()
    xs = solution['t']
    ys = solution['sol']
    ss = solution['types']


    xs = np.array(xs)
    ys = np.array(ys)
    ss = np.array(ss)
  
    # Reference solution
    reftable = "eq237.txt"
    refarray = np.genfromtxt(reftable, delimiter=',')
    ls = refarray[:,0]
    ytrue = refarray[abs(ls -l) < 1e-8, 1]
    yerr = np.abs((ytrue - ys[-1])/ytrue)
    print(yerr)

    # Write LaTeX table to file
    round_to_n = lambda n, x: x if x == 0 else round(x, - int(math.floor(math.log10(abs(x)))) + (n-1))
    steps = ss.shape[0] - 1
    steps_wkb = np.count_nonzero(ss)
    steps_rk = steps - steps_wkb 
    n_fevals = (w_count + g_count)/N
    outputf = "riccati-paper/tables/bremer237_oscode.tex"
    outputpath = Path(outputf)
    outputpath.touch(exist_ok = True)
    print("time/s: ", (end - start)*1e-9)
    runtime = (end - start)*1e-9/N
    evaltime = 0.0 #TODO
    lines = ""
    if os.stat(outputf).st_size != 0:
        with open(outputf, 'r') as f:
            lines = f.readlines()
    with open(outputf, 'w') as f:
        if lines == "": 
            f.write("\\begin{tabular}{l c c c c c c c}\n")
            f.write("\hline \hline \n$n$  &  $\max|\Delta u/u|$  &  $t_{\mathrm{solve}}$/\si{\s}  &  $t_{\mathrm{eval}}$/\si{\s}  &  $n_{\mathrm{s,WKB}}$  &  $n_{\mathrm{s,RK}}$  &  $n_{\mathrm{s,tot}}$  &  $n_{\mathrm{f}}$  \\\\ \hline\n")
        for line in lines:
            f.write(line)
        f.write("$10^{}$  &  ${}$  &  ${}$  &  ${}$  &  ${}$  &  ${}$  &  ${}$  &  ${}$  \\\\ \n".format(int(np.log10(l)), num2tex(round_to_n(3, max(yerr))), num2tex(round_to_n(3, runtime)), evaltime, steps_wkb, steps_rk, steps, round(n_fevals)))
        f.write("\hline \hline\n\end{tabular}\n")


def cosmology():
    
    # Inflaton mass, Planck mass, infl potential, spatial curvature
    m = 1
    mp = 1
    nv = 2
    K = 0
    
    # Cosmological background evolution
    def V(phi):
        """ inflationary potential"""
        return 0.5*m**2*phi**nv

    def dV(phi):
        """ derivative of the inflationary potential """
        return 0.5*nv*m**2*phi**(nv-1)
    
    def bgeqs(t, y):
        """ System of equations describing the evolution of the cosmological
        background """
        dy = np.zeros(y.shape)
        dy[0] = 4.0 + np.exp(y[0])*(4.0*K - 2.0*np.exp(2.0*t)*V(y[1]))
        dy[1] = - np.sqrt(6.0 + np.exp(y[0])*(6.0*K -
        2.0*np.exp(2.0*t)*V(y[1])))
        return dy
    
    def endinfl(t, y):
        """ Crosses zero when inflation ends """
        dphi = bgeqs(t,y)[1]
        epsilon = 0.5*dphi**2
        return epsilon - 1.

    ok_i = 2.1e-3
    N_i = 1.
    # Nominal end point of integration (we'll stop at the end of inflation)
    N_f = 80.
    # Points at which we'll obtain the background solution
    Nbg = 100000 # This determines grid fineness, see note below.
    N = np.linspace(N_i,N_f,Nbg)
    # Initial conditions
    phi_i = np.sqrt(4.*(1./ok_i + K)*np.exp(-2.0*N_i)/m**2)
    logok_i = np.log(ok_i)
    y_i = np.array([logok_i, phi_i])
    # Solve for the background until the end of inflation
    endinfl.terminal = True
    endinfl.direction = 1
    bgsol = solve_ivp(bgeqs, (N_i,N_f), y_i, events=endinfl, t_eval=N, rtol=1e-8, atol=1e-10)
    Nend = bgsol.t_events[0][0]
    Ntot = Nend - N_i
    print("Number of e-folds of inflation: ", Ntot)
    assert(Ntot > 55)

    logok = bgsol.y[0]
    phi = bgsol.y[1]
    N = bgsol.t
    dphi = np.array([-np.sqrt(6.0 + np.exp(Logok)*(6.0*K -
        2.0*np.exp(2.0*t)*V(Phi))) for Logok,Phi,t in zip(logok,phi,N) ])
    dlogok = np.array([4.0 + np.exp(Logok)*(4.0*K - 2.0*np.exp(2.0*t)*V(Phi)) for Logok,Phi,t in zip(logok,phi,N) ])
    dE_E = dlogok - 4. -2.*dV(phi)*np.exp(logok)*np.exp(2.*N)/dphi
    E = 0.5*dphi**2
    # Damping term
    g = 0.5*(3 - E + dE_E)
    # frequency
    logw = 0.5*logok
    
    # Determining pivot wavevector
    # Find which index corresponds to having ~50 e-folds of inflation left
    icrossing = np.nonzero(np.where(N - N_i > 11.0, N, 0))[0][0]
    print(icrossing)
    print("Pivot wavevector crosses horizon at efolds: ", N[icrossing] - N_i)
    kpivot = np.exp(-logw[icrossing])
    # Observable range of perturbations is 1e4 -- 2e-1 Mpc^-1
    kmin = 1e-4/0.05*kpivot
    kmax = 2e-1/0.05*kpivot
    print("Comoving minimum and maximum wavenumber: ", kmin, kmax)
    
    # Get a crude estimate for derivative of gamma
    gp = np.array([(g[i] - g[i-1])/(N[i] - N[i-1]) for i in range(1, len(g))])

    # Frequency corresponding to smallest scale (largest k, \ell)
    w2_kmax = gp + np.exp(2*(logw + np.log(kmax)))[1:] - g[1:]**2


    # Find horizon crossing times
    # range of wavevectors
    ks = np.logspace(np.log10(kmin),np.log10(kmax),10)
    end = np.zeros_like(ks,dtype=int)
    endindex = 0
    for i in range(len(ks)):
        for j in range(endindex,Nbg):
            if np.exp(-0.5*logok[j])/ks[i] > 100:
                end[i] = j
                endindex = j
                break
    print("Horizon crossing times: ", N[end])

    # Individual mode solution for k = kmax, with original ODE and its damping-free form
    solkmax = oscode.solve(N, logw + np.log(kmax), g, N_i+0.01, N[end[-1]], 1.0, 0.0, even_grid = True, t_eval = N[100:end[-1]], logw = True)
    solkmaxtilde = oscode.solve(N[1:], np.sqrt(w2_kmax+0j), np.zeros_like(N[1:]), N_i+0.01, N[end[-1]], 1.0, 0.0, even_grid = True, t_eval = N[100:end[-1]])
    ucts = solkmax['x_eval']
    uctstilde = solkmaxtilde['x_eval']

    plt.style.use('riccatipaper')
    fig, ax = plt.subplots(2, 2, sharex = True, figsize = (6, 5))

    #ax[0,0].set_ylabel("$\ln\omega(N)$")
    #ax[0, 1].set_ylabel("$\gamma(N)$")
    #ax[1,0].set_ylabel("$\\tilde{\omega}^2 \equiv \gamma' + \omega^2 - \gamma^2$")
    ax[1,0].set_xlabel("$N$")
    #ax[0,1].set_xlabel("$N$")
    ax[1,1].set_xlabel("$N$")
    #ax[0,0].set_xlabel("$N$")
    ax[0,0].set_xlim(N[100], 18.0)
    ax[0,1].set_xlim(N[100], 18.0)
#    ax[1,1].set_ylim(-1e-7, 1e7)
    
    #ax[0,0].plot(N, logw, label = "Hubble horizon")
    ax[0,0].plot(N, np.zeros_like(N), '--', color='grey')
    ax[0,0].plot(N, np.exp(2*(logw + np.log(kmax))), label = '$\omega^2$', color='black')
    ax[0,0].plot(N, g, label = '$\gamma$')
    ax[0,1].plot(N, np.zeros_like(N), '--', color='grey')
    ax[0,1].plot(N[1:], gp + np.exp(2*(logw + np.log(kmax)))[1:] - g[1:]**2, label='$\\tilde{\omega}^2$', color='black')
    ax[1,0].plot(N[100:end[-1]], np.zeros_like(ucts), '--', color='grey')
    ax[1,0].plot(N[100:end[-1]], ucts, color='black', lw = 0.7)
    ax[1,1].plot(N[100:end[-1]], np.zeros_like(ucts), '--', color='grey')
    ax[1,1].plot(N[100:end[-1]], uctstilde, color='black', lw = 0.7)
   
    #ax[0].axhspan(-np.log(100*kmax), -np.log(kmin), color = 'black', alpha = 0.1, ec = None)
    #ax[0].axvspan(N_i, 17.5, color = 'red', alpha = 0.1, ec = None)
    #ax[1].axvspan(N_i, 17.5, color = 'red', alpha = 0.1, ec = None)
    #ax[2].axvspan(N_i, 17.5, color = 'red', alpha = 0.1, ec = None)
    
    ax[0,0].set_yscale('symlog')
    ax[0,1].set_yscale('symlog')
    ax[1,0].set_yscale('symlog', linthresh = 1e-5)
    ax[1,1].set_yscale('symlog')
    ax[0,1].set_yticks([-1e1, 0, 1e1, 1e3, 1e5, 1e7, 1e9, 1e11])
    ax[0,0].set_yticks([-1e1, 0, 1e1, 1e3, 1e5, 1e7, 1e9, 1e11])
    ax[1,0].set_yticks([-1e1, -1e-1, -1e-3, -1e-5, 0, 1e-5, 1e-3, 1e-1, 1e1])
    ax[1,1].set_yticks([-1e5, -1e3, -1e1, 0, 1e1, 1e3])
    ax[0,0].set_ylabel('Coefficients in the ODE')
    ax[1,0].set_ylabel('$\mathcal{R}_k$')
    ax[1,1].set_ylabel('$\\tilde{\mathcal{R}}_k$')

    ax[0,0].legend()
    ax[0,1].legend()
    fig.subplots_adjust(hspace=0)

#    plt.show()
    plt.tight_layout()
    plt.savefig('riccati-paper/plots/cosmology.pdf')



#for m in np.logspace(1, 9, num = 9):
#    legendre(m)
#residual()

#for m in np.logspace(1, 7, num = 7):
#    print("Testing solver on Bremer 2018 Eq. (237) with lambda = {}".format(m))
#    print("Low tolerance")
#    # Low tolerance
#    eps = 1e-12
#    epsh = 1e-13
#    n = 20
#    outdir = "/mnt/home/fagocs/riccati-paper/tables/new/"
#    if m < 1e7:
##        Bremer237(m, n, eps, epsh, outdir, rk = False, rdc = False, oscode = False, wkbmarching = True, kummer = False)
#        Bremer237(m, n, eps, epsh, outdir, rk = False, rdc = False, oscode = False, wkbmarching = False, kummer = True)
#    else:
##        Bremer237(m, n, eps, epsh, outdir, rk = False, rdc = False, oscode = False, wkbmarching = True, kummer = False)
#        Bremer237(m, n, eps, epsh, outdir, rk = False, rdc = False, oscode = False, wkbmarching = False, kummer = True)
#
#airy()
#for m in np.logspace(1, 7, num = 4):
#    print("starting ", m)
#    oscode237(m)
#    print("done ", m)
#cosmology()
#convergence()
bremer237_timing_fig("/mnt/home/fagocs/riccati-paper/tables/new")
#Bremer237(1e1)

