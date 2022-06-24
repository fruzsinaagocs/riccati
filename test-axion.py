import os
import sys
import numpy as np
import riccati
import matplotlib.pyplot as plt

from tqdm import tqdm

from scipy.integrate import ode

script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(script_path+"/../../../simple_example/")
from example_solutions import *

def test_solve_alp(thetai=2.0):
    w = lambda tau: 1.5*np.ones_like(tau)
    g = lambda tau: 0.75/tau

    alpth = lambda tau: theta(tau, thetai)
    alpdth = lambda tau: dtheta_dtau(tau, thetai)

    ti = 1e-5
    tf = 50.0
    yi = thetai
    dyi = 0

    eps = 1e-10
    epsh = 1e-12
    xs, ys, dys, ss, ps, types = riccati.solve(w, g, ti, tf, yi, dyi, eps = eps, epsh = epsh)
    xs = np.array(xs)
    ys = np.array(ys)
    types = np.array(types)
    ytrue = alpth(xs)
    ncts = 5000
    xcts = np.linspace(ti, xs[-1], ncts)
    ycts = alpth(xcts)
    yerr = np.abs((ytrue - ys))/np.abs(ytrue)
    
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

def test_solve_harmonic_qcd(thetai = 2.0):
    w = lambda tau: v_ma(-tau_osc*tau)*v_aux(-tau)
    g = lambda tau: -0.5*v_aux_f2(-tau_osc*tau)/tau

    ti = -50.0
    tf = -0.1
    yi = thetai
    dyi = 0

    def f(tau, y):
        tt = tau_osc*tau
        return [y[1], -aux_f2(tt)*y[1]/tau - (ma(tt)*aux(tau))**2 * y[0]]

    def jac(tau, y):
        tt = tau_osc*tau
        return [[0, 1], [-(ma(tt)*aux(tau))**2, -aux_f2(tt)/tau]]

    sol = ode(f, jac).set_integrator('dop853', nsteps=1000, rtol=1e-10, atol=1e-14)
    sol.set_initial_value([thetai, 0], -ti)
    sol.integrate(10)

    eps = 1e-6
    epsh = 1e-12
    xs, ys, dys, ss, ps, types = riccati.solve(w, g, ti, tf, yi, dyi, eps = eps, epsh = epsh)
    xs = np.array(xs)
    xs_fl = np.flip(np.log10(-xs))
    ys = np.array(ys)
    ys_fl = np.flip(ys)
    types = np.flip(np.array(types))

    cond = -np.array(xs) > 0.3
    num_sol = np.array([[-t_new]+list(sol.integrate(t_new)) for t_new in tqdm(-np.array(xs)[cond])])
    yerr = np.abs((num_sol[:,1] - ys[cond]))/np.abs(num_sol[:,1])

    fig, ax = plt.subplots(3, 1, sharex=True)
    ax[0].plot(np.flip(np.log10(-num_sol[:,0])), np.flip(num_sol[:,1]), 'k-', lw=2, label='scipy')
    ax[0].plot(xs_fl[types==0], ys_fl[types==0], '.', color='C1', label='Nonosc step')
    ax[0].plot(xs_fl[types==1], ys_fl[types==1], '.', color='C0', label='Osc step')
    ax[1].semilogy(np.flip(np.log10(-xs)[cond]), np.flip(yerr), '.-', color='black')
    ax[2].semilogy(xs_fl, np.flip(v_ma(-xs*tau_osc)), label=r'$m$', c='b')
    ax[2].semilogy(xs_fl, np.flip(3.0*v_hubble(-xs*tau_osc)), label=r'$H$', c='b', ls='--')
    ax[2].semilogy(xs_fl, np.flip(-6.0*g(xs)), label=r'$\gamma$', c='r', ls='--')
    ax[2].semilogy(xs_fl, np.flip(w(xs)), label=r'$\omega$', c='r')
    ax[2].axvline(np.log10(143.7e6/tau_osc), c='k')
    ax[2].axvline(np.log10(143.7e6/tau_osc), c='k', ls='--')
    ax[2].legend()
    ax[0].set_ylabel("$y(x)$")
    ax[1].set_ylabel("Relative error")
    ax[2].set_ylabel(r"$\gamma$, $\omega$, $m$")
    ax[2].set_xlabel("$x$")
    ax[0].legend()
    ax[0].set_xlim([np.amin(xs_fl),1])
    plt.show()


if __name__ == "__main__":
    #test_solve_alp()
    test_solve_harmonic_qcd()
