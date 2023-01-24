.. _installation:

Installation
============

Before installation, all you'll need is Python and `numpy <https://numpy.org>`_. 

Via package managers
--------------------

We recommend using the Python package manager `pip <http://www.pip-installer.org/>`_:

.. code-block:: bash

    python -m pip install -U pip
    pip install -U riccati

or `conda <https://conda.io>`_

.. code-block:: bash

    conda update conda
    conda install -c conda-forge riccati

From source
-----------

Should you wish to use the development version, you can clone the source
repository and install as follows

.. code-block:: bash

    python -m pip install -U pip
    python -m pip install -U setuptools
    git clone https://github.com/fruzsinaagocs/riccati
    cd riccati
    python -m pip install -e .

Testing the installation
------------------------

You can run the unit tests to make sure the installation went OK.

To do that, install from source as above, and install `pytest
<https://docs.pytest.org>`_. Then execute the tests via

.. code-block:: bash
    
   python -m pip install -U pytest
   python -m pytest -v riccati/tests

If you didn't get any errors (warnings are OK), then the package is ready for
use.
