Module reference
================
Each module encapsulates a unit of functionality in the parameter estimation project.
The following is a brief description of each module and its most important classes, functions, and variables.

.. note::

   Some downloaded data and results of long-running functions are saved on disk in the :file:`cache` directory, which will be created as needed.
   However, there is no logic for updating the cache if the code is modified, which means cache files may need to be occassionally manually deleted.

Package init file
-----------------
Source code: :ghlink:`jetscape_stat/__init__.py`

.. automodule:: jetscape_stat

.. autodata:: systems

.. autodata:: keys

.. autodata:: labels

.. autodata:: ranges

.. autodata:: design_array

.. autodata:: data_list

.. autodata:: exp_data_list

.. autodata:: exp_cov

.. autodata:: observables

.. autofunction:: parse_system

Design
------
Source code: :ghlink:`jetscape_stat/design.py`

.. automodule:: jetscape_stat.design

.. autofunction:: generate_lhs

.. autoclass:: Design
   :members:

Emulator
--------
Source code: :ghlink:`jetscape_stat/emulator.py`

.. automodule:: jetscape_stat.emulator

.. autoclass:: Emulator
   :members:

MCMC
----
Source code: :ghlink:`jetscape_stat/mcmc.py`

.. automodule:: jetscape_stat.mcmc

.. autoclass:: Chain(path=Path('mcmc/chain.hdf'))
   :members:
   :exclude-members: map

.. autofunction:: credible_interval

Plots and figures
-----------------
Source code: :ghlink:`jetscape_stat/plots.py`

.. automodule:: jetscape_stat.plots

.. autofunction:: posterior

.. autofunction:: observables_design

.. autofunction:: observables_posterior

.. autofunction:: diag_emu

.. autofunction:: design

.. autofunction:: gp

