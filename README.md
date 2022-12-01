# ARDC: adaptive Riccati defect correction solver

## About

`riccati` is a `Python` package for solving ODEs of the form

$$ u''(t) + 2\gamma(t)u'(t) + \omega(t)u(t) = 0,$$

on some solution interval $t \in [t_0, t_1]$, and with initial conditions $u(t_0) = u_0$, $u'(t_0) = u'_0$.

`riccati` uses the adaptive Riccati defect correction method -- it switches between using nonoscillatory (spectral Chebyshev) and a specialised oscillatory solver (Riccati defect correction) to propagate the numerical solution based on its behaviour. More details here: ...

## Installation

To install:

```bash
git clone github.com/fruzsinaagocs/better-phase-fun
cd better-phase-fun
pip install .[all]
```

To run the unit tests:
```bash 
pytest riccati/tests/*
```

## Documentation

```bash
cd docs/
make html
```

Then open `docs/build/html/index.html` in your browser.
