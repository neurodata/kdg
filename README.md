`kdg` (Kernel Density Graph) is a package for exploring and using kernel density algorithms developed by the [neurodata group](https://neurodata.io).

Install
=======

Below we assume you have the default Python environment already configured on
your computer and you intend to install ``kdg`` inside of it.  If you want to
create and work with Python virtual environments, please follow instructions
on `venv <https://docs.python.org/3/library/venv.html>`_ and `virtual
environments <http://docs.python-guide.org/en/latest/dev/virtualenvs/>`_. We
also highly recommend conda. For instructions to install this, please look
at
`conda <https://docs.conda.io/projects/conda/en/latest/user-guide/install/>`_.

First, make sure you have the latest version of ``pip`` (the Python package
manager) installed. If you do not, refer to the `Pip documentation
<https://pip.pypa.io/en/stable/installing/>`_ and install ``pip`` first.


Install from Github
-------------------
You can manually download ``kdg`` by cloning the git repo master version and
running the ``setup.py`` file. That is, unzip the compressed package folder
and run the following from the top-level source directory using the Terminal::

    $ git clone https://github.com/jdey4/kdg
    $ cd kdg
    $ python3 setup.py install

Or, alternatively, you can use ``pip``::

    $ git clone https://github.com/jdey4/kdg
    $ cd kdg
    $ pip install .

Python package dependencies
---------------------------
``kdg`` requires the following packages:

- scikit-learn>=0.22.0
- scipy==1.4.1
- numpy==1.19.2

Hardware requirements
---------------------
``kdg`` package requires only a standard computer with enough RAM to support
the in-memory operations. GPU's can speed up the networks which are powered by 
tensorflow's backend. 

OS Requirements
---------------
This package is supported for all major operating systems. 

- **Linux**: Ubuntu 16.04
- **macOS**: Mojave (10.14.1)
- **Windows**: 10
