import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import riccati
import mpmath
import scipy.special as sp 


def airy():
    w = lambda x: np.sqrt(x)
    g = lambda x: np.zeros_like(x)
    xi = 1e0
    xf = 1e8
    eps = 1e-12
    epsh = 1e-13
    yi = sp.airy(-xi)[0] + 1j*sp.airy(-xi)[2]
    dyi = -sp.airy(-xi)[1] - 1j*sp.airy(-xi)[3]
    xs, ys, dys, ss, ps, stypes = riccati.solve(w, g, xi, xf, yi, dyi, eps = eps, epsh = epsh)
    xs = np.array(xs)
    ys = np.array(ys)
    ss = np.array(ss)
    ps = np.array(ps)
    stypes = np.array(stypes)

    xcts = np.logspace(np.log10(xi), np.log10(100), 10000)
    ycts = np.array([sp.airy(-x)[0] for x in xcts])
    ytrue = np.array([mpmath.airyai(-x) + 1j*mpmath.airybi(-x) for x in xs])
    yerr = np.abs((ytrue - ys)/ytrue)
    hs = np.array([xs[i] - xs[i-1] for i in range(1, xs.shape[0])])
    wiggles = np.array([sum(ps[:i+1])/(2*np.pi) for i in range(ps.shape[0])])

    tab20c = matplotlib.cm.get_cmap('tab20c')
    blue1 = tab20c.colors[0]
    blue2 = tab20c.colors[1]
    grey2 = tab20c.colors[-3]
    
    plt.style.use('sab') 
#    fig, ax = plt.subplots(2, 2)
    plt.figure()
    plt.title('Numerical solution of $u'' + tu = 0$')
    # Numerical solution
    plt.semilogx(xcts, ycts, color='black', label='analytic solution')
    plt.semilogx(xs[stypes==0], ys[stypes==0], '.', color='C0', label='Chebyshev step')
    plt.semilogx(xs[stypes==1], ys[stypes==1], '.', color='C1', label='Riccati step')
    plt.ylabel('$u(t)$')
    plt.xlim((1, 100))
    plt.legend()
#     # Stepsize
#     ax[0,1].loglog(xs[:-1], hs, color=blue1)
#     ax[0,1].loglog(np.logspace(0,8,10), np.logspace(0,8,10), '--', color=blue2)
#     ax[0,1].set_title('Stepsize')
#     ax[0,1].set_ylabel('Stepsize, $h$')
#     ax[0,1].annotate('$\propto t$', (5e3, 1e4), rotation=30)
#     ax[0,1].set_xlim((1e0, 1e8))
#     # Error
#     ax[1,0].loglog(xs[ss==True], yerr[ss==True], color=blue1)
#     ax[1,0].loglog(xs, eps*np.ones_like(xs), '--', color=grey2, label='$\epsilon$')
#     ax[1,0].loglog(xs[1:], wiggles*np.finfo(float).eps, '--', color=blue2,\
#             label='$K\cdot \epsilon_{\mathrm{mach}}$')
#     ax[1,0].annotate('minimum error $=K\cdot \epsilon_{\mathrm{mach}}$', (3e3, 5e-14))
#     ax[1,0].annotate('$K=$ condition number (# of periods)', (3e3, 5e-15))
#     ax[1,0].annotate('$\epsilon_{\mathrm{mach}}=$ machine precision', (3e3, 5e-16))
#     ax[1,0].set_title('Numerical error')
#     ax[1,0].set_xlabel('$t$')
#     ax[1,0].set_ylabel('Relative error, $|\Delta y/y|$')
#     ax[1,0].set_xlim((xi, xf))
#     ax[1,0].legend()
#     # Wiggles
#     ax[1,1].loglog(xs[1:], ps/(2*np.pi))
#     ax[1,1].set_ylabel('Periods traversed per step') 
#     ax[1,1].set_xlabel('$t$') 
#     ax[1,1].set_title('Oscillations per step')
#     ax[1,1].loglog(np.logspace(0,8,10), np.logspace(0,8*3/2,10), '--', color=blue2)
#     ax[1,1].annotate('$\propto t^{\\frac{3}{2}}$', (5e3, 1.1e6), rotation=35)
#     ax[1,1].set_xlim((1e0, 1e8))
# #    plt.show()
    plt.savefig("SAB-airy-numsol.pdf")

def burst():
    m = int(40)
    w = lambda x: np.sqrt(m**2 - 1)/(1 + x**2)
    g = lambda x: np.zeros_like(x)
    xi = -m
    xf = m
    eps = 1e-3
    epsh = 1e-13
    bursty = lambda x: np.sqrt(1 + x**2)/m*(np.cos(m*np.arctan(x)) + 1j*np.sin(m*np.arctan(x))) 
    burstdy = lambda x: 1/np.sqrt(1 + x**2)/m*((x + 1j*m)*np.cos(m*np.arctan(x))\
            + (-m + 1j*x)*np.sin(m*np.arctan(x)))
    yi = bursty(xi)
    dyi = burstdy(xi)
    xs, ys, dys, ss, ps, stypes = riccati.solve(w, g, xi, xf, yi, dyi, eps = eps, epsh = epsh)
    xs = np.array(xs)
    ys = np.array(ys)
    stypes = np.array(stypes)

    xcts1 = np.linspace(-m, -m/10, 500)
    xcts2b = np.logspace(-8, np.log10(m/10), 1000)
    xcts2a = -1*xcts2b[::-1]
    xcts3 = np.linspace(m/10, m, 500)
    xcts = np.concatenate((xcts1, xcts2a, xcts2b, xcts3))
    ycts = np.array([bursty(x) for x in xcts])

    tab20c = matplotlib.cm.get_cmap('tab20c')
    blue1 = tab20c.colors[0]
    blue2 = tab20c.colors[1]
    grey2 = tab20c.colors[-3]
    
    plt.style.use('cmotalk') 
    fig, ax = plt.subplots(2, 2)
    plt.title('Numerical solution')
    # Numerical solution
    ax[0,0].plot(xcts, ycts, color='black', label='analytic solution')
    ax[0,0].plot(xs[stypes==0], ys[stypes==0], '.', color='C0', label='Chebyshev step')
    ax[0,0].plot(xs[stypes==1], ys[stypes==1], '.', color='C1', label='Riccati step')
    ax[0,0].set_title('Numerical solution')
    ax[0,0].set_ylabel('$y(t)$')
    ax[0,0].set_xlim((-m/2, m/2))
    ax[0,0].set_ylim((-0.4, 0.2))
    ax[0,0].legend()

    c = tab20c.colors
    for i, mi in enumerate([1e6, 1e5, 1e4, 1e3]):
        
        m = mi
        w = lambda x: np.sqrt(m**2 - 1)/(1 + x**2)
        g = lambda x: np.zeros_like(x)
        xi = -m
        xf = m
        eps = 1e-14*mi
        epsh = 1e-10
        bursty = lambda x: np.sqrt(1 + x**2)/m*(np.cos(m*np.arctan(x)) + 1j*np.sin(m*np.arctan(x))) 
        burstdy = lambda x: 1/np.sqrt(1 + x**2)/m*((x + 1j*m)*np.cos(m*np.arctan(x))\
                + (-m + 1j*x)*np.sin(m*np.arctan(x)))
        yi = bursty(xi)
        dyi = burstdy(xi)
        xs, ys, dys, ss, ps, stypes = riccati.solve(w, g, xi, xf, yi, dyi, eps = eps, epsh = epsh)
        xs = np.array(xs)
        ys = np.array(ys)
        ss = np.array(ss)
        ps = np.array(ps)
        stypes = np.array(stypes)
       
        ytrue = np.array([bursty(x) for x in xs])
        yerr = np.abs((ytrue - ys)/ytrue)
        hs = np.array([xs[i] - xs[i-1] for i in range(1, xs.shape[0])])
        wiggles = np.array([sum(ps[:i+1])/(2*np.pi) for i in range(ps.shape[0])])
        # Rescale xs:
        xs = xs/mi
    
        # Stepsize
        ax[0,1].semilogy(xs[:-1], hs, color=c[i], label=r'$n=10^{}$'.format(int(np.log10(mi))))
    #    ax[0,1].semilogy(np.logspace(0,8,10), np.logspace(0,8,10), '--', color=blue2)
        ax[0,1].set_title('Stepsize')
        ax[0,1].set_ylabel('Stepsize, $h$')
        ax[0,1].set_xscale('symlog', linthresh=1e-5)
        ax[0,1].legend()
    #    ax[0,1].annotate('$\propto x$', (5e3, 1e4), rotation=30)
    #    ax[0,1].set_xlim((1e0, 1e8))
        # Error
        ax[1,0].semilogy(xs[ss==True], yerr[ss==True], color=c[i])
        ax[1,0].semilogy(xs, eps*np.ones_like(xs), '--', color=c[i+16], label='$\epsilon$')
#        ax[1,0].semilogy(xs[1:], wiggles*np.finfo(float).eps, '--', color=c[i+4],\
#                label='$K\cdot \epsilon_{\mathrm{mach}}$')
#        ax[1,0].annotate('minimum error $=K\cdot \epsilon_{\mathrm{mach}}$', (3e3, 5e-14))
#        ax[1,0].annotate('$K=$ condition number (# of wiggles)', (3e3, 5e-15))
#        ax[1,0].annotate('$\epsilon_{\mathrm{mach}}=$ machine precision', (3e3, 5e-16))
        ax[1,0].set_title('Numerical error')
        ax[1,0].set_xlabel('$t/n$')
        ax[1,0].set_ylabel('Relative error, $|\Delta y/y|$')
        #ax[1,0].set_xscale('symlog', linthresh=1e-5)
        ax[1,0].set_xlim((-1, 1))
        ax[1,0].set_ylim((1e-13, 1e-6))
    #    ax[1,0].legend()
        # Wiggles
        ax[1,1].semilogy(xs[1:], ps/(2*np.pi), color=c[i])
        ax[1,1].set_xscale('symlog', linthresh=1e-5)
        ax[1,1].set_ylabel('Periods traversed per step') 
        ax[1,1].set_xlabel('$t/n$') 
        ax[1,1].set_title('Oscillations per step')
    #    ax[1,1].semilogy(np.logspace(0,8,10), np.logspace(0,8*3/2,10), '--', color=blue2)
    #    ax[1,1].annotate('$\propto x^{\\frac{3}{2}}$', (5e3, 1.1e6), rotation=35)
        ax[1,1].set_xlim((-0.025, 0.025))
    #    plt.show()
    plt.savefig("burst-numsol-newswitching.pdf")

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
    c = tab20c.colors
    cols = [c[0], c[2], c[8], c[10]]

    plt.style.use('sab') 
    #fig, ax = plt.subplots(len(ks), 1, figsize=(2.3, 3.3), sharex=True)

    # Visualisation of num sol
    # for i, k in enumerate(ks):
    #     xs, ys, x1, y1, err = riccati.osc_step(w, g, x0, h, y0, dy0, epsres = eps, plotting = True, k=k)
    #     if i==0:
    #         ax[i].plot(xcts, ycts, color='black', label='analytic solution')
    #         ax[i].plot(xs, ys, '--', color=orange1, label='Riccati series')
    #         ax[i].legend(prop={'size': 3}, loc='upper left')
    #         ax[i].set_title("Solution of the Airy equation after $k$ Riccati iterations")
    #     else:
    #         ax[i].plot(xcts, ycts, color='black')
    #         ax[i].plot(xs, ys, '--', color=orange1)
    #     if i==len(ks)-1:
    #         ax[i].set_xlabel('$t$')
    #     ax[i].set_xlim((x0, 12))
    #     ax[i].set_ylim((-0.55, 0.55))
    #     ax[i].annotate('$k={}$'.format(k), (11, -0.5))
    # fig.text(0.0, 0.5, '$y(t)$', va='center', rotation='vertical')
    # plt.subplots_adjust(wspace=0, hspace=0)
    # plt.savefig('residual.pdf')
    
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
    xcoords = [10.82, 5.65, 3.97, 3.22]
    angles = [-45, -63, -68, -75]

    plt.figure()
    plt.xlabel("$k$")
    plt.ylabel("$\max_{t \in (t_i, t_i+h)} R\left[x_k(t)\\right]$")
    for j, m, xc, angle in zip(range(len(ms)), ms, xcoords, angles):
        m = m
        w = lambda x: np.sqrt(m**2 - 1)/(1 + x**2)

        for i, k in enumerate(ks):
            xs, ys, x1, y1, err = riccati.osc_step(w, g, x0, h, y0, dy0, epsres = eps, plotting = True, k=k)
            errs[i] = err
            bursty = lambda x: np.sqrt(1 + x**2)/m*(np.cos(m*np.arctan(x)) + 1j*np.sin(m*np.arctan(x))) 
            burstdy = lambda x: 1/np.sqrt(1 + x**2)/m*((x + 1j*m)*np.cos(m*np.arctan(x))\
                   + (-m + 1j*x)*np.sin(m*np.arctan(x)))
            y0 = bursty(x0)
            dy0 = burstdy(x0)   

        plt.semilogy(ks, errs, '.-', label='$\omega(t_i)={}$'.format(m), color=cols[j])
        #plt.semilogy(ks, 10.0**(np.log10(m)*(1.0-ks)), '--', color=c[j+4])
        #plt.annotate('$\propto \omega(t_i)^{-k}$', (xc, 1e-11), rotation=angle, color=tab20c.colors[j+4])
    
    plt.title('Maximum residual after $k$ Riccati iterations')
    plt.xlim((0,16))
    plt.ylim((1e-16, 9e7))
    plt.legend()
    plt.savefig("SAB-residual-k.pdf")


airy()
#burst()
#residual()
