from __future__ import annotations

import logging
import os
import re
from typing import TYPE_CHECKING

import openmc

if TYPE_CHECKING:
    from f4enix.input.libmanager import LibManager
    from f4enix.input.materials import Material, SubMaterial, Zaid


class OpenMCInputFiles:
    def __init__(self, path: str, name=None) -> None:
        """Class to handle the OpenMC input file generation

        Parameters
        ----------
        path : str
            path to input files
        name : str, optional
            name associated with the input file, by default None

        Returns
        -------
        None
        """
        self.path = path
        self.name = name
        self.initialise(path)

    def initialise(self, path: str) -> None:
        """Initialise OpenMC input files from xml

        Parameters
        ----------
        path : str
            path to input file destination

        Returns
        -------
        None
        """
        files = os.listdir(path)
        if ("geometry.xml" in files) and ("materials.xml") in files:
            self.load_geometry(
                os.path.join(path, "geometry.xml"),
                materials=os.path.join(path, "materials.xml"),
            )
        else:
            self.geometry = openmc.Geometry()
            self.materials = openmc.Materials()
        if "settings.xml" in files:
            self.load_settings(os.path.join(path, "settings.xml"))
        if "tallies.xml" in files:
            self.load_tallies(os.path.join(path, "tallies.xml"))
        else:
            self.tallies = openmc.Tallies()

    def load_geometry(self, geometry: str, materials: str) -> None:
        """Initialise OpenMC geometry from xml

        Parameters
        ----------
        geometry : str
            path to geometry input xml
        materials : str
            path to materials input xml, or openmc.Materials object

        Returns
        -------
        None
        """
        self.geometry = openmc.Geometry.from_xml(geometry, materials)

    def load_settings(self, settings: str) -> None:
        """Initialise OpenMC settings from xml

        Parameters
        ----------
        settings : str
            path to settings input xml

        Returns
        -------
        None
        """
        self.settings = openmc.Settings.from_xml(settings)

    def load_tallies(self, tallies: str) -> None:
        """Initialise OpenMC tallies from xml

        Parameters
        ----------
        talllies : str
            path to geometry input xml

        Returns
        -------
        None
        """
        self.tallies = openmc.Tallies.from_xml(tallies)

    def add_stopCard(self, nps: int, batches: int = 100) -> None:
        """Add number of particles to simulate

        Parameters
        ----------
        nps : int
            number of particles to simulate
        batches : int, optional
            number of batches, by default 100

        Returns
        -------
        None
        """
        if nps < batches:
            batches = nps

        self.settings.batches = batches
        self.settings.particles = int(nps / batches)

    def zaid_to_openmc(
        self, zaid: Zaid, openmc_material: openmc.Material, libmanager: LibManager
    ) -> None:
        """Convert Zaid to OpenMC format with atom or weight fraction

        Parameters
        ----------
        zaid : str
            Zaid to be added to the OpenMC material
        openmc_material : openmc.Material
            An instance of the OpenMC Material class representing the material used in the simulation.

        Returns
        -------
        None
        """
        nuclide = zaid.get_fullname(libmanager).replace("-", "")
        if zaid.fraction < 0.0:
            openmc_material.add_nuclide(nuclide, 100 * abs(zaid.fraction), "wo")
        else:
            openmc_material.add_nuclide(nuclide, 100 * abs(zaid.fraction), "ao")

    def submat_to_openmc(
        self,
        submaterial: SubMaterial,
        openmc_material: openmc.Material,
        libmanager: LibManager,
    ) -> None:
        """Handle submaterials in OpenMC

        Parameters
        ----------
        submaterial : SubMaterial
            An instance of the SubMaterial class representing a subcomponent of a material.
            It contains elements, each of which has ZAIDs (nuclide identifiers).
        openmc_material : openmc.Material
            An instance of the OpenMC Material class representing the material used in the simulation.
        libmanager : libmanager.LibManager
            An instance of the LibManager class responsible for managing external libraries.

        Returns
        -------
        None
        """
        for elem in submaterial.elements:
            for zaid in elem.zaids:
                self.zaid_to_openmc(zaid, openmc_material, libmanager)

    def mat_to_openmc(self, material: Material, libmanager: LibManager) -> None:
        """Convert a material to an OpenMC material and handle its submaterials.

        Parameters
        ----------
        material : Material
            An instance of the Material class representing the material to be converted.
        libmanager : LibManager
            An instance of the LibManager class responsible for managing external libraries.

        Returns
        -------
        None
        """
        matid = int(re.sub("[^0-9]", "", str(material.name)))
        matname = str(material.name)
        matdensity = abs(material.density)
        if material.density < 0:
            density_units = "g/cc"
        openmc_material = openmc.Material(matid, name=matname)
        openmc_material.set_density(density_units, matdensity)
        if material.submaterials is not None:
            for submaterial in material.submaterials:
                self.submat_to_openmc(submaterial, openmc_material, libmanager)
        self.materials.append(openmc_material)

    def matlist_to_openmc(self, matlist: list, libmanager: LibManager) -> None:
        """Convert a list of materials to OpenMC materials and load the geometry.

        Parameters
        ----------
        matlist : list of Material
            A list of Material instances to be converted. Each material should have the necessary
            attributes required by the mat_to_openmc method.
        libmanager : LibManager
            An instance of the LibManager class responsible for managing external libraries.

        Returns
        -------
        None
        """
        for material in matlist:
            self.mat_to_openmc(material, libmanager)
        self.load_geometry(os.path.join(self.path, "geometry.xml"), self.materials)

    def write(self, path: str) -> None:
        """Write OpenMC input files to xml

        Parameters
        ----------
        path : str
            path to input file destination

        Returns
        -------
        None
        """
        self.geometry.export_to_xml(os.path.join(path, "geometry.xml"))
        self.settings.export_to_xml(os.path.join(path, "settings.xml"))
        self.tallies.export_to_xml(os.path.join(path, "tallies.xml"))
        self.materials.export_to_xml(os.path.join(path, "materials.xml"))


class OpenMCStatePoint:
    def __init__(self, spfile_path: str | os.PathLike) -> None:
        """Class for handling OpenMC tatepoint file

        Parameters
        ----------
        spfile_path : str
            path to statepoint file
        """
        self.initialise(spfile_path)
        self.version = self.read_openmc_version()

    def initialise(self, spfile_path: str) -> None:
        """Read in statepoint file

        Parameters
        ----------
        spfile_path : str
            path to statepoint file
        """
        try:
            # Retrieve the version from the statepoint file (convert from tuple of integers to string)
            self.statepoint = openmc.StatePoint(spfile_path)
            self.tally_numbers = []
            self.tally_comments = []
            for _, tally in self.statepoint.tallies.items():
                self.tally_numbers.append(tally.id)
                self.tally_comments.append(tally.name)
        except (FileNotFoundError, KeyError):
            logging.warning(
                "OpenMC version not found in the statepoint file for %s",
                spfile_path,
            )
            return None

    def read_openmc_version(self) -> str:
        """Get OpenMC version from statepoint file

        Returns
        -------
        str
            OpenMC version
        """
        version = ".".join(map(str, self.statepoint.version))
        return version

    def _get_tally_data(self, tally: openmc.Tally):
        """Extract tally data from statepoint file

        Parameters
        ----------
        tally : openmc.Tally
            openmc tally

        Returns
        -------
        df : pd.DataFrame
            pandas dataframe containing tally data
        """
        df = tally.get_pandas_dataframe()
        # df.to_csv('tally_'+str(tally.id)+'.csv')
        return df

    def tallies_to_dataframes(self):
        """Call to extract tally data from statepoint file

        Returns
        -------
        list
            list of rows with all sphere case tally data
        """
        tallies = {}
        for _, tally in self.statepoint.tallies.items():
            tallies[tally.id] = self._get_tally_data(tally)
        return tallies


class OpenMCSphereStatePoint(OpenMCStatePoint):
    def __init__(self, spfile_path: str) -> None:
        """Class to handle the data extraction of the Sphere leakage benchmark in OpenMC

        Parameters
        ----------
        spfile_path : str
            path to statepoint file
        """
        super().__init__(spfile_path)

    def _get_tally_data(self, rows: list, filter: openmc.Filter) -> list:
        """Extract tally data from statepoint file

        Parameters
        ----------
        rows : list
            list of rows to append tally data to
        filter : openmc.Filter
            determines OpenMC tally type

        Returns
        -------
        list
            list of rows with tally data appended
        """
        tally = self.statepoint.get_tally(filters=[filter])
        tally_n = tally.id
        tally_description = tally.name.title()
        energy_bins = tally.find_filter(openmc.EnergyFilter).values[1:]
        fluxes = tally.mean.flatten()
        errors = tally.std_dev.flatten()
        for energy, flux, error in zip(energy_bins, fluxes, errors):
            rows.append([tally_n, tally_description, energy, flux, error])
        return rows

    def tally_to_rows(self) -> list:
        """Call to extract tally data from statepoint file

        Returns
        -------
        list
            list of rows with all sphere case tally data
        """
        rows = []
        neutron_particle_filter = openmc.ParticleFilter(["neutron"])
        rows = self._get_tally_data(rows, neutron_particle_filter)
        photon_particle_filter = openmc.ParticleFilter(["photon"])
        rows = self._get_tally_data(rows, photon_particle_filter)
        return rows
