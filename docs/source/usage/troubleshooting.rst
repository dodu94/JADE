###############
Troubleshooting
###############

Unable to post-process a library
================================

It may happen that the request to post-process a library fails, prompting to
video that the library in question has not been run.

.. image:: /img/troubleshooting/notrun.png

This may puzzle the user who is sure to have run the library instead.

This may happen if:

* in the :ref:`mainconfig` file, not all the benchmark that are currently set 
  as active for the post-processing have been actually run;
* MCNP was run for the benchmarks but there were errors in the Monte Carlo
  simulations. JADE considers a benchmark to have been "run" only if a mctal or
  fmesh file has been produced for each of the MCNP run foreseen by the benchmark.


JADE fails on first library assessment
======================================

A ``FileNotFound`` error may be encountered the first time an assessment is
requested using JADE. Most likely, this would be caused by the impossibility
to call the MCNP executable directly from a cmd prompt.
Indeed, by default, MCNP installation provide also an ad hoc prompt where the
necessary enviroment variables are already set up and does not set such variables
as user enviroment variables.

To solve this issue it is sufficient to set-up/modify the following enviromental
variables for the user account:

DATAPATH
    path to the folder containing the xsdir files, it will be something like
    ``<Custom path to MCNP folder>\MCNP_DATA\``.
PATH
    the path to the MCNP executables needs to be added to the PATH enviromental
    variables. The path will be similar to
    ``<Custom path to MCNP folder>\MCNP_CODE\bin``.
