.. highlight:: shell

Installation
============

**Copulas** can be installed in two ways:

* From stable release
* From source

Stable Release
--------------

To install **Copulas**, run the following command in your terminal `pip`:

.. code-block:: console

    pip install copulas

This is the preffered method to install **Copulas**, as it will always install the most recent
and stable release.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

From source
-----------

The source code of **Copulas** can be downloaded from the `Github repository`_

You can clone the repository and install with the following command in your terminal:

.. code-block:: console

    git clone git://github.com/sdv-dev/Copulas
    cd Copulas
    make install

If you are installing **Copulas** in order to modify its code, the installation must be done
from its sources, in the editable mode, and also including some additional dependencies in
order to be able to run the tests and build the documentation. Instructions about this process
can be found in the `Contributing guide`_.

.. _Contributing guide: ../contributing.html#get-started
.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/
.. _Github repository: https://github.com/sdv-dev/Copulas
