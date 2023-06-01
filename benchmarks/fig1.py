import numpy as np
import riccati
import scipy.special as sp
from scipy.integrate import solve_ivp
import matplotlib
from matplotlib import pyplot as plt
import math
import os
from pathlib import Path
import time
import pyoscode
from matplotlib.legend_handler import HandlerTuple
import pandas
import subprocess
from matplotlib.ticker import LogLocator


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

def Bremer237(l, n, eps, epsh, outdir, rdc = True, wkbmarching = False,\
              kummer = False, oscode = False, rk = False):
    """
    Solves problem (237) from Bremer's "On the numerical solution of second
    order ordinary differential equations in the high-frequency regime" paper.
    """
    def w(x):
        return l*np.sqrt(1 - x**2*np.cos(3*x))

    def g(x):
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
    reftable = "./data/eq237.txt"
    refarray = np.genfromtxt(reftable, delimiter=',')
    ls = refarray[:,0]
    ytrue = refarray[abs(ls -l) < 1e-8, 1]
    errref = refarray[abs(ls -l) < 1e-8, 2]

    if rdc:
        print("riccati")
        N = 1000 # Number of repetitions for timing
        epsh = epsh
        n = n
        p = n
        start = time.time_ns()
        info = riccati.solversetup(w, g, n = n, p = p)
        for i in range(N):
            xs, ys, dys, ss, ps, stypes = riccati.solve(info, xi, xf, yi, dyi, eps = eps, epsh = epsh, hard_stop = True)
        end = time.time_ns()
        ys = np.array(ys)
        # Compute statistics
        runtime = (end - start)*1e-9/N
        yerr = np.abs((ytrue - ys[-1])/ytrue)
        # Write to txt file
        # Create dir
        outputf = outdir + "bremer237-rdc.txt" 
        outputpath = Path(outputf)
        outputpath.touch(exist_ok = True)
        lines = ""
        if os.stat(outputf).st_size != 0:
            with open(outputf, 'r') as f:
                lines = f.readlines()
        with open(outputf, 'w') as f:
            if lines == "": 
                f.write("# method, l, eps, relerr, tsolve, errlessref, params\n")
            for line in lines:
                f.write(line)
            f.write("{}, {}, {}, {}, {}, {}, {}".format("rdc",\
                    l, eps, round_to_n(3, max(yerr)), round_to_n(3, runtime),\
                    (yerr < errref)[0], "(n = {}; p = {}; epsh = {})".format(n, p, epsh)))
            f.write("\n")

    if oscode:
        print("oscode")
        if eps < 1e-8 and l < 1e4:
            N = 10
        else:
            N = 100
        # Time this process
        start = time.time_ns()
        for i in range(N):
            solution = pyoscode.solve_fn(w, g, xi, xf, yi, dyi, rtol = eps)
        end = time.time_ns()
        ys = solution['sol']
        ys = np.array(ys)
        yerr = np.abs((ytrue - ys[-1])/ytrue)
        runtime = (end - start)*1e-9/N
        # Write to txt file
        outputf = outdir + "bremer237-oscode.txt" 
        outputpath = Path(outputf)
        outputpath.touch(exist_ok = True)
        lines = ""
        if os.stat(outputf).st_size != 0:
            with open(outputf, 'r') as f:
                lines = f.readlines()
        with open(outputf, 'w') as f:
            if lines == "": 
                f.write("# method, l, eps, relerr, tsolve, errlessref, params\n")
            for line in lines:
                f.write(line)
            f.write("{}, {}, {}, {}, {}, {}, {}".format("oscode",\
                    l, eps, round_to_n(3, max(yerr)), round_to_n(3, runtime),\
                    (yerr < errref)[0], "(nrk = default; nwkb = default)"))
            f.write("\n")

    if kummer:
        print("Kummer phase function method")
        kummerscript = "test_eq237"
        # Compile and run fortran code with command-line arguments l and eps
        os.chdir("./ext-codes/Phase-functions")
        subprocess.run(["make", "clean"])
        subprocess.run(["make", kummerscript])
        kummeroutput = subprocess.run(["./{}".format(kummerscript), str(l), str(eps)], capture_output = True)
        kummerstdout = kummeroutput.stdout.decode('utf-8')
        print(kummerstdout)
        y, runtime, n_fevals = [float(i) for i in kummerstdout.split()]
        yerr = np.abs((ytrue - y)/ytrue)
        # Write to txt file
        os.chdir("../../")
        outputf = outdir + "bremer237-kummer.txt" 
        outputpath = Path(outputf)
        outputpath.touch(exist_ok = True)
        lines = ""
        if os.stat(outputf).st_size != 0:
            with open(outputf, 'r') as f:
                lines = f.readlines()
        with open(outputf, 'w') as f:
            if lines == "": 
                f.write("# method, l, eps, relerr, tsolve, errlessref, params\n")
            for line in lines:
                f.write(line)
            f.write("{}, {}, {}, {}, {}, {}, {}".format("kummer",\
                    l, eps, round_to_n(3, max(yerr)), round_to_n(3, runtime),\
                    (yerr < errref)[0], "()"))
            f.write("\n")
    
    if wkbmarching:
        print("WKB marching method")
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
        outputf = "\\\"" +  outdir + "bremer237-wkbmarching.txt\\\""
        outputpath = Path(outputf)
        os.chdir("./ext-codes/adaptive-WKB-marching-method")
        # Run matlab script (will write file)
        os.system("matlab -batch \"global aeval; {}({}, {}, {}, {}); exit\" ".format(matlabscript, l, eps, N, outputf))
        os.chdir("../../")

    if rk:
        print("Runge--Kutta")
        # We're only running this once because it's slow
        atol = 1e-14
        method = "DOP853"
        f = lambda t, y: np.array([y[1], -l**2*(1 - t**2*np.cos(3*t))*y[0]])
        time0 = time.time_ns()
        sol = solve_ivp(f, [-1, 1], [0, l], method = method, rtol = eps, atol = atol)
        time1 = time.time_ns()
        runtime = (time1 - time0)*1e-9
        err = np.abs((sol.y[0,-1]- ytrue)/ytrue)[0]
        # Write to txt file
        outputf = outdir + "bremer237-rk.txt" 
        outputpath = Path(outputf)
        outputpath.touch(exist_ok = True)
        lines = ""
        if os.stat(outputf).st_size != 0:
            with open(outputf, 'r') as f:
                lines = f.readlines()
        with open(outputf, 'w') as f:
            if lines == "": 
                f.write("# method, l, eps, relerr, tsolve, errlessref, params\n")
            for line in lines:
                f.write(line)
            f.write("{}, {}, {}, {}, {}, {}, {}".format("rk",\
                    l, eps, round_to_n(3, err), round_to_n(3, runtime),\
                    (err < errref)[0], "(atol = {}; method = {})".format(atol, method)))
            f.write("\n")


def joss_fig(outdir):

    # Example solution

    def f(t, y):
        yp = np.zeros_like(y)
        yp[0] = y[1]
        yp[1] = -l**2*(1 - t**2*np.cos(3*t))*y[0]
        return yp

    eps = 1e-4
    atol = 1e-14
    method = "DOP853"
    l = 1e2
    f = lambda t, y: np.array([y[1], -l**2*(1 - t**2*np.cos(3*t))*y[0]])
    t_eval = np.linspace(-1, 1, 5000)
    sol = solve_ivp(f, [-1, 1], [0, l], method = method, rtol = eps, atol = atol, t_eval = t_eval)



    # Helper function
    round_to_n = lambda n, x: x if x == 0 else round(x, - int(math.floor(math.log10(abs(x)))) + (n-1))

    # Read in little tables and combine into one pandas dataframe
    outputfs = [outdir + "bremer237-{0}.txt".format(method) for method in ["rk", "rdc", "kummer", "oscode", "wkbmarching"]] 
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
    fig, (ax1, ax0) = plt.subplots(1, 2, figsize = (6, 3))
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

    l = ax0.legend([(l1, l6), (l2, l7), (l3, l8), (l4, l9), (l5, l10), (l11, l12, l13, l14, l15), (l16, l17, l18, l19, l20)], ['DOP853 (SciPy)', '\\texttt{oscode}', 'WKB marching', "Kummer's phase function", 'ARDC (\\texttt{riccati})', '$\\varepsilon = 10^{-12}$', '$\\varepsilon = 10^{-6}$'], handler_map = {tuple: HandlerTupleVertical()})

    ax0.set_ylabel('runtime/s, $t_{\mathrm{solve}}$')
    ax0.set_xlim((1e1, 1e7))
    ax0.set_ylim((2e-4, 2e5))
    ax0.set_xlabel('$\lambda$')
    ax0.yaxis.set_minor_locator(LogLocator(numticks=15, subs=np.arange(-4, 6)))

    ax1.plot(t_eval, sol.y[0,:], color='black', lw = 0.5)
    ax1.set_xlabel('$t$')
    ax1.set_ylabel('$u(t)$')
    ax1.set_xlim((-1, 1))

    plt.savefig('./timing-fig-2.pdf')



outdir = os.getcwd() + "/data/"
epss, epshs, ns = [1e-12, 1e-6], [1e-13, 1e-9], [35, 20]
for m in np.logspace(1, 7, num = 7):
    print("Testing solver on Bremer 2018 Eq. (237) with lambda = {}".format(m))
    for eps, epsh, n in zip(epss, epshs, ns):
        if m < 1e7:
            Bremer237(m, n, eps, epsh, outdir, rk = True, rdc = True,\
                      oscode = True, wkbmarching = True, kummer = True)
        else:
            Bremer237(m, n, eps, epsh, outdir, rk = False, rdc = True,\
                      oscode = True, wkbmarching = True, kummer = True)
joss_fig(outdir)



