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
    affiliation: 1
  - name: Alex Barnett 
    affiliation: 1
affiliations: 
  - name: Center for Computational Mathematics, Flatiron Institute, 162 Fifth Avenue, New York, 10010 NY, USA 
    index: 1
date: 3 February 2023
bibliography: paper.bib

---

# Summary

Highly oscillatory ordinary differential equations (ODE)s pose a computational
challenge for standard solvers available in scientific computing libraries as
these typically have runtimes scaling as $\mathcal{O}(\omega)$, with $\omega$
being the oscillation frequency. 

The `riccati`
(Python) package implements the efficient numerical method described in @ardc
(dubbed ARDC for adaptive Riccati defect correction) for solving ODEs of the
form
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

`oscode` [@oscode-joss; @oscode-theory] and the WKB-marching method[^1]
@wkbmarching1; @wkbmarching2 are examples of low-order adaptive oscillatory
solvers, efficient when no more than about 6 digits of accuracy are required or $\omega(t)$ is near-constant.
A high-order alternative is the Kummer's phase function-based method by
@kummerphase; @phasefntp[^2], whose current implementation supports solving
\autoref{eq:ode} in the highly oscillatory regime. Other existing numerical methods are
reviewed in e.g. @petzoldrev. @autoref{fig:solver-comparison} compares the
performance of the above specialised solvers and one of Scipy's [@scipy] built-in methods [@dop853]
by plotting their runtime against the frequency parameter $\lambda$ while
solving
\begin{equation}\label{eq:runtime-ode}
u'' + \omega^2(t) u = 0, \text{ where } \omega^2(t) = \lambda^2(1 - t^2\cos 3t ),
\end{equation}
on the interval $t \in [-1, 1]$, subject to the initial conditions $u(-1) = 0$,
$u'(-1) = \lambda$. The runtimes were measured at two settings of the required
relative tolerance $\epsilon$, $10^{-6}$ and $10^{-12}$. The figure
demonstrates the advantage `riccati''s adaptiveorder provides at low tolerances
as it avoids the runtime-increase `oscode' and the WKB marching methods exhibit
at low-to-intermediate frequencies, and confirms that its runtime is virtually
independent of the oscillation frequency. 

![Performance comparison of `riccati' (labelled ARDC) against state-of-the-art oscillatory solvers: `oscode', the WKB marching method, Kummer's phase function method, and a high-order Runge--Kutta method (RK78) [@dop853] on \autoref{eq:runtime-ode} with a varying frequency parameter $\lambda$. Solid and dashed lines denote runs with a relative tolerance settings of $\varepsilon = 10^{-12}$ and $10^{-6}$, respectively. \label{fig:solver-comparison}](solver-comparison-timing.pdf)


[^1]: Available from https://github.com/JannisKoerner/adaptive-WKB-marching-method
[^2]: Available from https://github.com/JamesCBremerJr/Phase-functions

# Acknowledgements
 
We thank Jim Bremer, Charlie Epstein, Manas Rachh, and Leslie Greengard for
useful discussions. The Flatiron Institute is a division of the Simons
Foundation.

# References
