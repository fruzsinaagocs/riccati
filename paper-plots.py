import numpy as np
import riccati
import scipy.special as sp
from scipy.integrate import solve_ivp
import matplotlib
#matplotlib.use("Qt5Agg")
from matplotlib import pyplot as plt
import math
import mpmath
import os
from num2tex import num2tex
from pathlib import Path
import time
import cProfile, pstats
import pyoscode as oscode

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
    xs, ys, dys, ss, ps, stypes, statdict = riccati.solve(w, g, xi, xf, yi, dyi, eps = eps, epsh = epsh, n = 32, p = 32)
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
    wiggles = np.array([sum(ps[:i+1])/(2*np.pi) for i in range(ps.shape[0])])

    tab20c = matplotlib.cm.get_cmap('tab20c')
    blue1 = tab20c.colors[0]
    blue2 = tab20c.colors[1]
    grey2 = tab20c.colors[-3]
    
    plt.style.use('riccatipaper') 
    fig, ax = plt.subplots(2, 2, figsize=(6, 6))
    # Numerical solution
    ax[0,0].semilogx(xcts, ycts, color='black', lw = 0.7, label='analytic solution')
    ax[0,0].semilogx(xs[stypes==0], ys[stypes==0], '.', color='C0', label='Chebyshev step')
    ax[0,0].semilogx(xs[stypes==1], ys[stypes==1], '.', color='C1', label='Riccati step')
    ax[0,0].set_ylabel('$R\left[u(t)\\right]$')
    ax[0,0].set_xlim((1, 65))
    ax[0,0].legend()
    # Stepsize
    ax[0,1].loglog(xs[:-1], hs, color='black')
    ax[0,1].loglog(np.logspace(0,8,10), np.logspace(0,8,10), '--', color=blue2)
    ax[0,1].set_ylabel('Stepsize, $h$')
    ax[0,1].annotate('$\propto t$', (5e3, 2e3), rotation=45, c=blue1)
    ax[0,1].set_xlim((1e0, 1e8))
    # Error
    ax[1,0].loglog(xs[ss==True], yerr[ss==True], color='black')
    ax[1,0].loglog(xs, eps*np.ones_like(xs), '--', color=grey2, label='$\\varepsilon$')
    ax[1,0].loglog(xs[1:], wiggles*np.finfo(float).eps, '--', color=blue2,\
            label='$K\cdot \\varepsilon_{\mathrm{mach}}$')
    ax[1,0].set_xlabel('$t$')
    ax[1,0].set_ylabel('Relative error, $|\Delta u/u|$')
    ax[1,0].set_xlim((xi, xf))
    ax[1,0].legend()
    # Wiggles
    ax[1,1].loglog(xs[1:], ps/(2*np.pi), color='black')
    ax[1,1].set_ylabel('$n_{\mathrm{osc}}$ per step') 
    ax[1,1].set_xlabel('$t$') 
    ax[1,1].loglog(np.logspace(0,8,10), np.logspace(0,8*3/2,10), '--', color=blue2)
    ax[1,1].annotate('$\propto t^{\\frac{3}{2}}$', (5e3, 2e6), rotation=45, c=blue1)
    ax[1,1].set_xlim((1e0, 1e8))
#    plt.show()
    plt.savefig("riccati-paper/plots/airy-numsol.pdf")





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
            f.write("\\begin{tabular}{l c c c c c c c c c c}\n")
            f.write("\hline \hline \n$n$  &  $\max|\Delta u/u|$  &  $t_{\mathrm{solve}}$/\si{\s}  &  $t_{\mathrm{eval}}$/\si{\s}  &  $n_{\mathrm{s,Ricc}}$  &  $n_{\mathrm{s,Cheb}}$  &  $n_{\mathrm{s,tot}}$  &  $n_{\mathrm{f}}$  &  $n_{\mathrm{LS}}$  &  $n_{\mathrm{LU}}$  &  $n_{\mathrm{sub}}$ \\\\ \hline\n")
        for line in lines:
            f.write(line)
        f.write("$10^{}$  &  ${}$  &  ${}$  &  ${}$  &  ${}$  &  ${}$  &  ${}$  &  ${}$  &  ${}$  &  ${}$  &  ${}$\\\\ \n".format(int(np.log10(m)), num2tex(round_to_n(3, max(yerr))), num2tex(round_to_n(3, runtime)), evaltime, steps_ricc, steps_cheb, steps, round(n_fevals), n_LS, n_LU, n_sub))
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
    fig, ax = plt.subplots(len(ks), 1, figsize=(6, 4), sharex=True)

    # Visualisation of num sol
    stats = riccati.Stats()
    n = 16
    for i, k in enumerate(ks):
        xs, ys, x1, y1, err = riccati.osc_step(w, g, x0, h, y0, dy0, stats, n = n, epsres = eps, plotting = True, k=k)
        if i==0:
            ax[i].plot(xcts, ycts, color='black', label='analytic solution')
            ax[i].plot(xs, ys, '--', color=orange1, label='Riccati series')
            ax[i].legend(prop={'size': 3}, loc='upper left')
#            ax[i].set_title("Solution of the Airy equation after $k$ Riccati iterations")
        else:
            ax[i].plot(xcts, ycts, color='black')
            ax[i].plot(xs, ys, '--', color=orange1)
        if i==len(ks)-1:
            ax[i].set_xlabel('$x$')
        ax[i].set_xlim((x0, 12))
        ax[i].set_ylim((-0.55, 0.55))
        ax[i].annotate('$k={}$'.format(k), (11, -0.5))
    fig.text(0.0, 0.5, '$y(x)$', va='center', rotation='vertical')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig('riccati-paper/plots/residual.pdf')
    
    # Just residual 
    N = 16
    ks = np.linspace(0, N, N+1, dtype=int)
    ms = [int(1e1), int(1e2), int(1e3), int(1e4)]
    g = lambda x: np.zeros_like(x)
    x0 = 0
    h = 2.0
    eps = 1e-12
    epsh = 1e-13
    errs = np.zeros(N+1)
    xcoords = [11.05, 5.66, 3.97, 3.13]
    angles = [-np.arctan(np.log10(mi)*1/2.5)/np.pi*180 for mi in ms]
#    symbols = ['.', '^', '+', 'o']

    plt.figure(figsize = (6, 4))
    plt.xlabel("$k$")
    plt.ylabel("$R\left[x^{(k)}\\right]$")
    for j, m, xc, angle in zip(range(len(ms)), ms, xcoords, angles):
        m = m
        w = lambda x: np.sqrt(m**2 - 1)/(1 + x**2)

        for i, k in enumerate(ks):
            xs, ys, x1, y1, err = riccati.osc_step(w, g, x0, h, y0, dy0, stats, epsres = eps, plotting = True, k=k)
            errs[i] = err
            bursty = lambda x: np.sqrt(1 + x**2)/m*(np.cos(m*np.arctan(x)) + 1j*np.sin(m*np.arctan(x))) 
            burstdy = lambda x: 1/np.sqrt(1 + x**2)/m*((x + 1j*m)*np.cos(m*np.arctan(x))\
                   + (-m + 1j*x)*np.sin(m*np.arctan(x)))
            y0 = bursty(x0)
            dy0 = burstdy(x0)   

        plt.semilogy(ks, errs, '.-', label='$\omega(t_i) = 10^{}$'.format(int(np.log10(m))), color=tab20c.colors[j*4])
        plt.semilogy(ks, 10.0**(np.log10(m)*(1.0-ks)), '--', color=tab20c.colors[j*4+1])
        plt.annotate('$\propto \omega(t_i)^k$' , (xc, 1e-11), rotation=angle, color=tab20c.colors[j*4])
    
#    plt.title('Maximum residual after $k$ Riccati iterations')
    plt.xlim((0,16))
    plt.ylim((1e-16, 2e8))
    plt.legend()
    plt.savefig("riccati-paper/plots/residual-k.pdf")

def Bremer237(l):
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

    eps = 1e-12 #, n = p = 16
    epsh = 1e-13
    yi = 0.0
    dyi = l
    yi_vec = np.array([yi, dyi])
    N = 100
    n = 40
    p = 40
#    n = 32
#    p = 32

    # Time this process
    start = time.time_ns()
    for i in range(N):
        xs, ys, dys, ss, ps, stypes, statdict = riccati.solve(w, g, xi, xf, yi, dyi, eps = eps, epsh = epsh, n = n, p = p, hard_stop = True)
    end = time.time_ns()
    xs = np.array(xs)
    ys = np.array(ys)
    ss = np.array(ss)
    ps = np.array(ps)
    stypes = np.array(stypes)
    print(statdict)
  
    # Reference solution
    reftable = "eq237.txt"
    refarray = np.genfromtxt(reftable, delimiter=',')
    ls = refarray[:,0]
    ytrue = refarray[abs(ls -l) < 1e-8, 1]
    yerr = np.abs((ytrue - ys[-1])/ytrue)
    print(eps, epsh, yerr)

    # Write LaTeX table to file
    round_to_n = lambda n, x: x if x == 0 else round(x, - int(math.floor(math.log10(abs(x)))) + (n-1))
    steps = (statdict["cheb steps"][0] + statdict["ricc steps"][0], statdict["cheb steps"][1] + statdict["ricc steps"][1])
    steps_ricc = statdict['ricc steps']
    steps_cheb = statdict['cheb steps']
    n_fevals = (w_count + g_count)/N
    n_LS = statdict["linear solves"]
    n_LU = statdict["LU decomp"]
    n_sub = statdict["substitution"]
    outputf = "riccati-paper/tables/bremer237.tex"
    outputpath = Path(outputf)
    outputpath.touch(exist_ok = True)
    print("time/s: ", (end - start)*1e-9/N)
    runtime = (end - start)*1e-9/N
    evaltime = 0.0 #TODO
    lines = ""
    if os.stat(outputf).st_size != 0:
        with open(outputf, 'r') as f:
            lines = f.readlines()
            lines = lines[:-2] # Remove trailing \hline-s and \end{tabular}
    with open(outputf, 'w') as f:
        if lines == "": 
            f.write("\\begin{tabular}{l c c c c c c c c c c}\n")
            f.write("\hline \hline \n$n$  &  $\max|\Delta u/u|$  &  $t_{\mathrm{solve}}$/\si{\s}  &  $t_{\mathrm{eval}}$/\si{\s}  &  $n_{\mathrm{s,Ricc}}$  &  $n_{\mathrm{s,Cheb}}$  &  $n_{\mathrm{s,tot}}$  &  $n_{\mathrm{f}}$  &  $n_{\mathrm{LS}}$  &  $n_{\mathrm{LU}}$  &  $n_{\mathrm{sub}}$ \\\\ \hline\n")
        for line in lines:
            f.write(line)
        f.write("$10^{}$  &  ${}$  &  ${}$  &  ${}$  &  ${}$  &  ${}$  &  ${}$  &  ${}$  &  ${}$  &  ${}$  &  ${}$\\\\ \n".format(int(np.log10(l)), num2tex(round_to_n(3, max(yerr))), num2tex(round_to_n(3, runtime)), evaltime, steps_ricc, steps_cheb, steps, round(n_fevals), n_LS, n_LU, n_sub))
        f.write("\hline \hline\n\end{tabular}\n")

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
            lines = lines[:-2] # Remove trailing \hline-s and \end{tabular}
    with open(outputf, 'w') as f:
        if lines == "": 
            f.write("\\begin{tabular}{l c c c c c c c}\n")
            f.write("\hline \hline \n$n$  &  $\max|\Delta u/u|$  &  $t_{\mathrm{solve}}$/\si{\s}  &  $t_{\mathrm{eval}}$/\si{\s}  &  $n_{\mathrm{s,WKB}}$  &  $n_{\mathrm{s,RK}}$  &  $n_{\mathrm{s,tot}}$  &  $n_{\mathrm{f}}$  \\\\ \hline\n")
        for line in lines:
            f.write(line)
        f.write("$10^{}$  &  ${}$  &  ${}$  &  ${}$  &  ${}$  &  ${}$  &  ${}$  &  ${}$  \\\\ \n".format(int(np.log10(l)), num2tex(round_to_n(3, max(yerr))), num2tex(round_to_n(3, runtime)), evaltime, steps_wkb, steps_rk, steps, round(n_fevals)))
        f.write("\hline \hline\n\end{tabular}\n")







#for m in np.logspace(1, 9, num = 9):
#    legendre(m)
#residual()
#for m in np.logspace(1, 7, num = 7):
#    print("starting with Riccati ", m)
#    Bremer237(m)
Bremer237(1e7)
#airy()
#for m in np.logspace(1, 7, num = 4):
#    print("starting ", m)
#    oscode237(m)
#    print("done ", m)


