import os
import sys
import numpy as np
import riccati
import matplotlib.pyplot as plt

script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(script_path+"/../../../simple_example/")
from example_solutions import theta, dtheta_dtau

def test_solve_alp():
    thetai = 2.0 # Initial field value

    w = lambda tau: 1.5*np.ones_like(tau)
    g = lambda tau: 0.75/tau

    alpth = lambda tau: theta(tau, thetai)
    alpdth = lambda tau: dtheta_dtau(tau, thetai)

    ti = 1e-5
    tf = 50.0
    yi = thetai
    dyi = 0

    eps = 1e-6
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


if __name__ == "__main__":
    test_solve_alp()
