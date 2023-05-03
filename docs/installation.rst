.. _installation:

Installation
============

Before installation, all you'll need is Python and `numpy <https://numpy.org>`_. 

Via package managers
--------------------

pip
~~~

You can install the package using the Python package manager `pip <http://www.pip-installer.org/>`_:

.. code-block:: bash

    python -m pip install -U pip
    pip install -U riccati

Optional dependencies can be installed by specifying one or more options in

.. code-block:: bash

    pip install -U riccati[docs,tests,all]

conda
~~~~~

You may also use `conda <https://conda.io>`_:

.. code-block:: bash

    conda update conda
    conda install -c conda-forge riccati

From source
~~~~~~~~~~~

Should you wish to use the development version, you can clone the source
repository and install as follows

.. code-block:: bash

    python -m pip install -U pip
    python -m pip install -U setuptools
    git clone https://github.com/fruzsinaagocs/riccati
    cd riccati
    python -m pip install -e .

Optional dependencies are installed the same way as in the `pip` installation:

.. code-block:: bash

    python -m pip install -e .[docs,tests,all]

Testing the installation
------------------------

You can run the unit tests to make sure the installation went OK.

Make sure you have the tests' dependencies (`pytest`, `scipy`, and `mpmath`,
see the optional dependencies under the pip install instructions or in `setup.py`).

If during the tests' run you didn't get any errors (warnings are OK), then the package is ready for
use.

If you installed from source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once you're in the installation directory, execute the tests via

.. code-block:: bash
    
   python -m pytest -v


If you used a package manager
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Change directory to the location of the installed package, which might be something like

- conda: `~/miniconda/envs/<env-name>/lib/python<3.n>/site-packages/riccati`, with `<env-name>` being the name of your conda environment, and `3.n` your Python version,
- virtualenv and pip: `<path-to-env>/lib/python<3.n>/site-packages/riccati` with `<path-to-env>` being the path to your virtual environment.

Then execute the tests:

.. code-block:: bash

    cd <location-of-riccati>
    python -m pytest -v


