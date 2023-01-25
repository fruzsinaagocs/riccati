riccati
=======

**riccati** is a Python package implementing the `adaptive Riccati defect
correction (ARDC) method <https://arxiv.org/abs/2212.06924>`_ by Agocs & Barnett. 

ARDC is a numerical method for solving ordinary diffential equations (ODEs) of the form

.. math::

    u''(t) + 2\gamma(t)u'(t) + \omega^2(t)u(t) = 0,

on some solution interval :math:`t \in [t_0, t_1]` and subject to the initial
conditions :math:`u(t_0) = u_0`, :math:`u'(t_0) = u'_0`.

This documentation will show you how to use the package which is under active
development on `GitHub <https://github.com/fruzsinaagocs/riccati>`_.

Shields

How to use the docs
-------------------

Start by following the (brief) :ref:`installation` guide.
After that you may get started straight away with the :ref:`quickstart`, or
check out some more :ref:`examples`. Each function in the module is documented
in the :ref:`api`.

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   examples
    

.. toctree::
   :maxdepth: 1
   :caption: Reference

   api

To generate offline documentation, get the source code as described in the
:ref:`installation` guide, and run

.. code-block:: bash
    
   cd docs/
   make html

This will create a `docs/build/` directory in which the `html/` subdirectory
contains these documentation pages in html format.

License and attribution
-----------------------

Copyright 2022-2023 The Simons Foundation, Inc. 

**riccati** is free software available under the Apache License 2.0, for
details see the `LICENSE <https://github.com/fruzsinaagocs/riccati/LICENSE>`_. 

If you use **riccati** in your work, please cite our paper (`arxiv <https://arxiv.org/abs/2212.06924>`_, `BibTex <https://ui.adsabs.harvard.edu/abs/2022arXiv221206924A/exportcitation>`_).

Contributing
------------

We welcome bug reports, patches, feature requests, and other comments via
`GitHub issues <https://github.com/fruzsinaagocs/riccati/issues>`_.




