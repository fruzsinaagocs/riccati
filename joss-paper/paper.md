---
title: 'riccati: an adaptive, spectral solver for oscillatory ODEs'
tags:
  - Python
  - numerical methods
  - ordinary differential equations
  - oscillatory problems
authors:
  - name: Fruzsina Julia Agocs
    orcid: 0000-0002-1763-5884
    affiliation: Center for Computational Mathematics, Flatiron Institute, 162
    Fifth Avenue, New York, 10010 NY, USA 
  - name: Alex Barnett 
    affiliation: Center for Computational Mathematics, Flatiron Institute, 162
    Fifth Avenue, New York, 10010 NY, USA 
date: 3 Febuary 2023
bibliography: paper.bib

---

# Summary

Highly oscillatory ordinary differential equations (ODE)s pose a computational
challenge for standard solvers available in scientific computing libraries as
these typically have runtimes scaling as $\mathcal{O}(\omega)$, with $\omega$
being the oscillation frequency. 

The `riccati`
(Python) package implements the efficient numerical method described in [@ardc] for solving ODEs of the form
\begin{equation}\label{eq:ode}
u''(t) + 2\gamma(t)u'(t) + \omega^2(t) u(t) = 0, \quad t \in [t_0, t_1],
\end{equation}
subject to the initial conditions $u(t_0) = u_0$, $u'(t_0) = u'_0$. The frequency $\omega(t)$
and friction $\gamma(t)$ terms are given smooth real-valued functions. The
solution $u(t)$ may vary between oscillatory and slowly-changing over the
integration range since `riccati` will adapt both its choice of method and
stepsize to achieve an $\mathcal{O}(1)$ (frequency-independent) runtime. The
solver is capable of producing _dense output_, i.e. can return a numerical
solution estimate at a pre-determined set of $t$-values, at the cost of a few
arithmetic operations per $t$-point.

# Statement of need

Some specialised numerical methods exist to solve \autoref{eq:ode} in
the high-frequency ($\omega >> 1$) regime, but out of those that have software implementations
none are both (1) adaptive, meaning that they stay efficient if the solution of
the ODE does not oscillate; and (2) high-order accurate, so that the user may
request many digits of accuracy without loss of efficiency. `riccati` fills
this gap as a high-order (spectral) adaptive solver.

`oscode` [@oscode-joss, @oscode-theory] and the WKB-marching method[^1]
[@wkbmarching1, @wkbmarching2] are examples of low-order adaptive oscillatory
solvers, efficient when no more than about 6 digits of accuracy are required or $\omega(t)$ is near-constant.
A high-order alternative is the Kummer's phase function-based method by
Bremer[^2] [@kummerphase, @phasefntp], whose current implementation supports solving
\autoref{eq:ode} in the highly oscillatory regime. Other existing numerical methods are
reviewed in e.g. [@petzoldrev]. 

![Caption!](solver-comparison-timing.pdf){ width = 50% }


[^1]: Available from https://github.com/JannisKoerner/adaptive-WKB-marching-method
[^2]: Available from https://github.com/JamesCBremerJr/Phase-functions

# Acknowledgements
 
We thank Jim Bremer, Charlie Epstein, Manas Rachh, and Leslie Greengard for
useful discussions. The Flatiron Institute is a division of the Simons
Foundation.

# References
