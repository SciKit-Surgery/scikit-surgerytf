scikit-surgerytf
===============================

.. image:: https://weisslab.cs.ucl.ac.uk/WEISS/SoftwareRepositories/SNAPPY/scikit-surgerytf/raw/master/project-icon.png
   :height: 128px
   :width: 128px
   :target: https://weisslab.cs.ucl.ac.uk/WEISS/SoftwareRepositories/SNAPPY/scikit-surgerytf
   :alt: Logo

.. image:: https://weisslab.cs.ucl.ac.uk/WEISS/SoftwareRepositories/SNAPPY/scikit-surgerytf/badges/master/build.svg
   :target: https://weisslab.cs.ucl.ac.uk/WEISS/SoftwareRepositories/SNAPPY/scikit-surgerytf/pipelines
   :alt: GitLab-CI test status

.. image:: https://weisslab.cs.ucl.ac.uk/WEISS/SoftwareRepositories/SNAPPY/scikit-surgerytf/badges/master/coverage.svg
    :target: https://weisslab.cs.ucl.ac.uk/WEISS/SoftwareRepositories/SNAPPY/scikit-surgerytf/commits/master
    :alt: Test coverage

.. image:: https://readthedocs.org/projects/scikit-surgerytf/badge/?version=latest
    :target: http://scikit-surgerytf.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status


Author: Matt Clarkson

scikit-surgerytf is part of the `SNAPPY`_ software project, developed at the `Wellcome EPSRC Centre for Interventional and Surgical Sciences`_, part of `University College London (UCL)`_.

scikit-surgerytf supports Python 3.6+, and tensorflow >= 2.0.0.

The aim of scikit-surgerytf is to provide a home for various Tensor Flow examples and
utilities and to show best practice. It's NOT meant to be a layer on-top of Tensor Flow
or provide a new kind-of platform. The aim is that researchers can learn from examples,
and importantly, learn how to deliver an algorithm that can be used by other people
out of the box, with just a ```pip install```, rather than a new user having to
re-implement stuff, or struggle to get someone else's code running. Researchers
can commit their research to this repository, or use the `PythonTemplate`_ to
generate their own project, and put it therein.

Features
----------

Each project herein should provide the following:

* Code that passes pylint.
* Unit testing, as appropriate.
* Sufficient logging, including date, time, software (git) version, runtime folder, machine name.
* A main class containing a network that can be run separately in train/test mode.
* Saving of learned network weights.
* Loading of pre-train weights and either continue training, or testing some input.
* Visualisation with TensorBoard.
* The ability to be run repeatedly for hyper-parameter tuning via python scripting, not bash.
* The ability to be callable from within a Jupyter Notebook, and thereby amenable to weekly writup's for supervisions.
* One or more command line programs that are pip-installable, providing a subsequent user to train and test your algorithm.


Developing
----------

Cloning
^^^^^^^

You can clone the repository using the following command:

::

    git clone https://weisslab.cs.ucl.ac.uk/WEISS/SoftwareRepositories/SNAPPY/scikit-surgerytf


Running tests
^^^^^^^^^^^^^
Pytest is used for running unit tests:
::

    pip install pytest
    python -m pytest


Linting
^^^^^^^

This code conforms to the PEP8 standard. Pylint can be used to analyse the code:

::

    pip install pylint
    pylint --rcfile=tests/pylintrc sksurgerytf


Installing
----------

You can pip install directly from the repository as follows:

::

    pip install git+https://weisslab.cs.ucl.ac.uk/WEISS/SoftwareRepositories/SNAPPY/scikit-surgerytf



Contributing
----------

Please see the `contributing guidelines`_.


Useful links
----------

* `Source code repository`_
* `Documentation`_


Licensing and copyright
-----------------------

Copyright 2019 University College London.
scikit-surgerytf is released under the Apache Software License 2.0. Please see the `license file`_ for details.


Acknowledgements
----------------

Supported by `Wellcome`_ and `EPSRC`_.


.. _`Wellcome EPSRC Centre for Interventional and Surgical Sciences`: http://www.ucl.ac.uk/weiss
.. _`source code repository`: https://weisslab.cs.ucl.ac.uk/WEISS/SoftwareRepositories/SNAPPY/scikit-surgerytf
.. _`Documentation`: https://scikit-surgerytf.readthedocs.io
.. _`SNAPPY`: https://weisslab.cs.ucl.ac.uk/WEISS/PlatformManagement/SNAPPY/wikis/home
.. _`University College London (UCL)`: http://www.ucl.ac.uk/
.. _`Wellcome`: https://wellcome.ac.uk/
.. _`EPSRC`: https://www.epsrc.ac.uk/
.. _`contributing guidelines`: https://weisslab.cs.ucl.ac.uk/WEISS/SoftwareRepositories/SNAPPY/scikit-surgerytf/blob/master/CONTRIBUTING.rst
.. _`license file`: https://weisslab.cs.ucl.ac.uk/WEISS/SoftwareRepositories/SNAPPY/scikit-surgerytf/blob/master/LICENSE
.. _`PythonTemplate`: https://weisslab.cs.ucl.ac.uk/WEISS/SoftwareRepositories/PythonTemplate