.. _simptokamak:

Simple Tokamak
--------------

.. figure:: /img/benchmarks/simp_tokamak_xz_2.jpg
    :width: 600
    :align: center

    Simple Tokamak geometry (xz)


.. figure:: /img/benchmarks/simp_tokamak_xy_2.jpg
    :width: 600
    :align: center

    Simple Tokamak geometry (xy)

The simplified tokamak is a 60 degree model with a representation of 
the major components of a tokamak. Owing to its simplicity, this model is well suited to transport code 
benchmarking while being representative of geometry and energy regimes in the nuclear 
analysis of fusion devices.

Geometry and run parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Reflecting planes are included on the lateral bounds of the 
sector to approximate the toroidal symmetry of the tokamak.  Typical materials for a 
tokamak are used in a homogenized form for the blanket, divertor, vacuum vessel,
ports, toroidal and poloidal field coils and the bioshield. In total, the model has
168 cells, 344 surfaces and 92 nuclides. The source term is an isotropic point source 
emitting neutrons with an energy of 14 MeV positioned in the center of the plasma
region of the geometry.

Tallies
^^^^^^^

The following nuclear responses are tallied in this benchmark:

Tally n. 24
    Neutron flux in blanket.
Tally n. 34
    Photon flux in blanket.
Tally n. 44
    Neutron flux in PF coils.
Tally n.54
    Photon flux in PF coils.
Tally n. 64
    Neutron spectrum (binned in Vitamin-J 175 energy groups) in outboard blanket.
Tally n. 74
    Photon spectrum (binned in Vitamin-J 175 energy groups) in outboard blanket.
Tally n. 84
    Neutron spectrum (binned in Vitamin-J 175 energy groups) in PF coils.
Tally n. 94
    Photon spectrum (binned in Vitamin-J 175 energy groups) in PF coils.
Tally n. 6
    Neutron nuclear heating in blanket.
Tally n. 16
    Photon nuclear heating in blanket.
Tally n. 26
    Neutron nuclear heating in PF coils.
Tally n. 36
    Photon nuclear heating in PF coils.
Tally n. 46
   Neutron nuclear heating in VV.
Tally n. 56
    Photon nuclear heating in VV.
Tally n. 104
    Tritium production in blanket.
Tally n. 114
    DPA in divertor cell.