"""
riccati
=======

riccati is a package for solving ODEs of the form

.. math:: u''(t) + 2\gamma(t)u'(t) + \omega^2(t)u(t) = 0

on a solution interval :math:`t \in [t_0, t_1]` and subject to initial
conditions :math:`u(t_0) = u_0`, :math:`u'(t_0) = u'_0`. 

The package implements the adaptive Riccati defect correction 
`ADRC <https://arxiv.org/abs/2212.06924>`_ method, which switches between using a
nonoscillatory (spectral Chebyshev) and a specialised oscillatory solver
(Riccati defect correction) to propagate the numerical solution based on its
behaviour.

Submodules
----------

riccati comes with the following submodules:
    chebutils
        Helper functions to perform numerical differentiation, integration,
        interpolation, etc. on a Chebyshev grid
    solversetup
        Functions to set up the ODE solve and store information about the run
    stepsize
        Stepsize selection functions
    step
        Functions that perform a single step within the ODE solve
    evolve
        Functions that evolve the numerical ODE solution over several steps

Documentation
-------------

For details, please see the `online documentation
<https://riccati.readthedocs.io>`_ or inspect the docstrings in `IPython`
after importing riccati, e.g. `help(riccati.evolve.solve)`.

"""

from riccati.evolve import solve, osc_evolve, nonosc_evolve
from riccati.step import nonosc_step, osc_step
from riccati.stepsize import choose_nonosc_stepsize, choose_osc_stepsize
from riccati.solversetup import solversetup

from . import _version
__version__ = _version.get_versions()['version']
__uri__ = "https://riccati.readthedocs.io"
__author__ = "F. J. Agocs and A. H. Barnett"
__email__ = "fagocs@flatironinstitute.org"
__license__ = "Apache 2.0"
__description__ = "A package implementing the adaptive Riccati defect correction (ARDC) method for solving oscillatory second order linear ODEs."
__all__ = ["evolve", "step", "stepsize", "solversetup", "chebutils"]
