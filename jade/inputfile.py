# -*- coding: utf-8 -*-

# Created on Mon Nov  4 17:21:24 2019

# @author: Davide Laghi

# Copyright 2021, the JADE Development Team. All rights reserved.

# This file is part of JADE.

# JADE is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# JADE is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with JADE.  If not, see <http://www.gnu.org/licenses/>.

import json
import os
import re
import sys
import textwrap
import warnings
from contextlib import contextmanager

from numjuggler import parser as par

import f4enix.input.materials as mat
from f4enix.input.d1suned import Reaction, ReactionFile


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def check_transport_activation(lib):
    # Operate on the newlib, should arrive in the 99c-31c format
    errmsg = """
 Please define the pair activation-transport lib for SDDR benchmarks
 (e.g. 99c-31c). See additional details on the documentation.
            """
    try:
        activationlib = lib.split("-")[0]
        transportlib = lib.split("-")[1]
    except IndexError:
        raise ValueError(errmsg)
    # Check that libraries have been correctly defined
    if activationlib + "-" + transportlib != lib:
        raise ValueError(errmsg)

    return activationlib, transportlib


class SerpentInputFile:
    def __init__(self, lines, materials, name=None):
        """
        Object representing a Serpent input file

        Parameters
        ----------
        cards : dic
            contains the cells, surfaces, settings and title cards.
        materials : matreader.MatCardList
            material list in the input.
        name : str, optional
            name associated with the file. The default is None.

        Returns
        -------
        None.

        """

        # All cards from parser separated by the materials
        self.lines = lines

        # Materials list (see matreader.py)
        self.materials = materials

        # Set a name
        self.name = name

    @classmethod
    def from_text(cls, inputfile):
        """
        Reads input file into list. Removes trailing empty lines.

        Parameters
        ----------
        cls : 'SerpentInputFile'
            The class itself.
        inputfile : path like object
            path to the Serpent input file.

        Returns
        -------
        SerpentInputFile instance with data from the input file.

        """
        with open(inputfile, "r") as f:
            lines = f.readlines()

        # Remove trailing empty lines
        idx = len(lines) - 1
        while True:
            if lines[idx].strip() == "":
                del lines[idx]
                idx -= 1
            else:
                break

        materials = None

        return cls(lines, materials, name=os.path.basename(inputfile).split(".")[0])

    def add_stopCard(self, nps: int) -> None:
        """Add number of particles card

        Parameters
        ----------
        nps : int
            number of particles to simulate
        """
        self.lines.append("\nset nps " + str(int(nps)) + "\n")

    def _to_text(self) -> str:
        """
        Get the input in Serpent formatted text

        Returns
        -------
        str
            Serpent formatted text for the input
        """

        # Add materials
        self.lines.append(self.materials.to_text())
        self.lines.append("\n")  # Missing

        toprint = ""
        for line in self.lines:
            toprint = toprint + line

        return toprint

    def write(self, out) -> None:
        """
        Write the input to a file

        Parameters
        ----------
        out : str
            path to the output file.

        Returns
        -------
        None.

        """
        to_print = self._to_text()

        with open(out, "w") as outfile:
            outfile.write(to_print)


'''
class OpenMCInputFiles:
    def __init__(
        self, geometry, settings, tallies, openmc_materials, materials, name=None
    ):
        """Object representing an OpenMC input file.

        Parameters
        ----------
        geometry : List[str], optional
            OpenMC geometry
        settings : List[str], optional
            OpenMC settings
        tallies : List[str], optional
            OpenMC tallies
        openmc_materials : List[str], optional
            OpenMC materials
        materials : List[str], optional
            material list in the input
        name : str, optional
            name associated with the file, by default None
        """

        # Geometry, settings and tallies for OpenMC
        self.geometry = geometry
        self.settings = settings
        self.tallies = tallies
        # Temporary material holder until materials supports openmc mats
        self.openmc_materials = openmc_materials

        # Materials list (see matreader.py)
        self.materials = materials

        # Set a name
        self.name = name

    # This should be updated to use xml elements rather than strings
    def _get_lines(self, path: str):
        """Read in lines from file.

        Parameters
        ----------
        path : str
            Path to file

        Returns
        -------
        Optional[List[str]]
            List of lines from the file or none if file not found.
        """
        if os.path.exists(path):
            with open(path, "r") as f:
                lines = f.readlines()
        else:
            lines = None
        return lines

    @classmethod
    def from_path(cls, path: str) -> "OpenMCInputFiles":
        """Reads contents of geometry, settings, tallies, openmc_materials from XML files.

        Parameters
        ----------
        path : str
            path to the files

        Returns
        -------
        OpenMCInputFiles
            Intance of class initialised with data from XML files.
        """

        geometry_file = os.path.join(path, "geometry.xml")
        geometry = cls._get_lines(cls, geometry_file)

        settings_file = os.path.join(path, "settings.xml")
        settings = cls._get_lines(cls, settings_file)

        tallies_file = os.path.join(path, "tallies.xml")
        tallies = cls._get_lines(cls, tallies_file)

        openmc_materials_file = os.path.join(path, "materials.xml")
        openmc_materials = cls._get_lines(cls, openmc_materials_file)

        materials = None

        name = os.path.basename(os.path.dirname(path))

        return cls(geometry, settings, tallies, openmc_materials, materials, name=name)

    def add_stopCard(self, nps: int) -> None:
        """Add number of particles to simulate

        Parameters
        ----------
        nps : int
            number of particles to simulate
        """
        for i, line in enumerate(self.settings):
            if "<settings>" in line:
                batches_line = "  <batches>100</batches>\n"
                self.settings.insert(i + 1, batches_line)
                particles = int(nps / 100)
                particles_line = "  <particles>" + str(particles) + "</particles>\n"
                self.settings.insert(i + 1, particles_line)
                break

    def _to_xml(self, libmanager) -> tuple:
        """Convert Class data to XML format

        Parameters
        ----------
        libmanager : libmanager
            Library manager

        Returns
        -------
        tuple
            A tuple containing XML representations of geometry, settings, tallies, and openmc_materials.
        """

        # Add materials
        self.openmc_materials = self.materials.to_xml(libmanager)

        geometry = ""
        for line in self.geometry:
            geometry = geometry + line

        settings = ""
        for line in self.settings:
            settings = settings + line

        tallies = ""
        for line in self.tallies:
            tallies = tallies + line

        openmc_materials = self.openmc_materials

        return geometry, settings, tallies, openmc_materials

    def write(self, path, libmanager) -> None:
        """
        Write the input to a file

        Parameters
        ----------
        out : str
            path to the output file.

        Returns
        -------
        None.

        """
        geometry, settings, tallies, openmc_materials = self._to_xml(libmanager)

        geometry_file = os.path.join(path, "geometry.xml")
        with open(geometry_file, "w") as outfile:
            outfile.write(geometry)

        settings_file = os.path.join(path, "settings.xml")
        with open(settings_file, "w") as outfile:
            outfile.write(settings)

        tallies_file = os.path.join(path, "tallies.xml")
        with open(tallies_file, "w") as outfile:
            outfile.write(tallies)

        materials_file = os.path.join(path, "materials.xml")
        with open(materials_file, "w") as outfile:
            outfile.write(materials)
'''
