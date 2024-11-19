##################
Default Benchmarks
##################
This section describes more in detail all the default benchmarks
that have been implemented in JADE, dividing them between
computational and experimental benchmarks. It is strongly recommended
that the user reads this documentation carefully before using a specific 
benchmark in JADE. 

.. important::
    * Not all benchmark inputs and related files can be distributed
      together with JADE due to licensing reasons. In this case, if the user
      provides evidence of licensing rights on specific benchmarks, the
      JADE team will provide the missing input which, once copied 
      in the ``<JADE_root>\Benchmark inputs`` folder, can be run in JADE. The user
      should follow the example of distributed benchmarks for folder and naming 
      convention.
    * For some of the benchmarks, weight windows (WW) have been produced and
      necessary for the benchmark run. Unfortunately, these WW files are often too
      heavy for them to be distributed with Git. These files must be requested 
      and inserted in to the same folder as the corresponding input file.  
    * The benchmarks included in JADE can be also divided between
      **transport** benchmarks and **activation** benchmarks (currently perfromed using
      D1S-UNED). It is recommended to run these two benchmarks
      categories separately, mostly because they require a different
      input in terms of the library to be assessed. If transport benchmarks
      expect a single library (e.g. ``31c``), activation ones require
      two: an activation library and a transport one for all zaids that
      cannot be activated (e.g. ``99c-31c``).
    * The irradiation and activation files if present for a SDDR benchmark should be 
      placed in the same folder as the input file. 
    * In activation benchmarks, the library that is considered the assessed one
      is always the activation library (i.e. the first provided). No track
      is kept during the post-processing of which was the transport library used
      and it is responsibility of the user to make sure that comparisons between
      activation libraries results are done in a coherent way. That is, **the
      same transport library should be always used**.


Overview
========
The following tables summarise the computational and experimental benchmarks
that are included in JADE **(I)**, the ones that are currently under development
**(D)**, and a future wishlist **(W)**. A more detailed description of the implemented 
benchmarks can be found in the following sections.

**COMPUTATIONAL BENCHMARKS:**

.. csv-table::
    :file: computational_overview.csv
    :widths: 70 10 10 10
    :header-rows: 1

**EXPERIMENTAL BENCHMARKS:**

.. csv-table::
    :file: exp_overview.csv
    :widths: 70 10 10 10
    :header-rows: 1

Computational Benchmarks
========================

.. include:: benchdesc/sphere.rst

.. include:: benchdesc/iter1d.rst

.. include:: benchdesc/tbm.rst

.. include:: benchdesc/cmodel.rst

.. include:: benchdesc/sphereSDDR.rst

.. include:: benchdesc/itercylSDDR.rst

.. include:: benchdesc/simptokamak.rst

Experimental Benchmarks
=======================

.. include:: benchdesc/fng/fng-sddr.rst

.. include:: benchdesc/fng/fng-blk.rst

.. include:: benchdesc/fng/fng-w.rst

.. include:: benchdesc/fng/fng-sic.rst

.. include:: benchdesc/fng/fng-hcpb.rst

.. include:: benchdesc/fng/fng-ss.rst

.. include:: benchdesc/fng/tud-fng.rst

.. include:: benchdesc/fng/tud-w.rst

.. include:: benchdesc/aspis/aspis-fe88.rst

.. include:: benchdesc/aspis/aspis-pca-replica.rst

.. include:: benchdesc/oktavian.rst

.. include:: benchdesc/tiara.rst

.. include:: benchdesc/fns-tof.rst

.. include:: benchdesc/tud-fe.rst
