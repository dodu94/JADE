# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 16:52:32 2019

@author: Davide Laghi

Copyright 2021, the JADE Development Team. All rights reserved.

This file is part of JADE.

JADE is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

JADE is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with JADE.  If not, see <http://www.gnu.org/licenses/>.
"""

from __future__ import annotations

import datetime
import logging
import os
import sys
from dataclasses import dataclass
from enum import Enum

import pandas as pd
import yaml

from jade.exceptions import fatal_exception
from jade.plotter import PlotType


class BinningType(Enum):
    ENERGY = "Energy"
    CELLS = "Cells"
    TIME = "Time"
    TALLY = "tally"
    DIR = "Dir"
    USER = "User"
    SEGMENTS = "Segments"
    MULTIPLIER = "Multiplier"
    COSINE = "Cosine"
    CORA = "Cor A"
    CORB = "Cor B"
    CORC = "Cor C"


class Configuration:
    def __init__(self, conf_file):
        """
        Parser of the main configuration file

        Parameters
        ----------
        conf_file : path like object
            path to configuration file.

        Returns
        -------
        None.

        """
        # ############ load conf file sheets ############
        self.conf_file = conf_file
        self.read_settings()

    def _process_path(self, file_path: str) -> str:
        """Process a file path to make sure that its absolute

        Parameters
        ----------
        file_path : str
            Absolute or None.

        Returns
        -------
        str
            An absolute file path.
        """
        if pd.isnull(file_path) is not True:
            if os.path.isabs(file_path) is not True:
                file_path = os.path.join(os.path.dirname(self.conf_file), file_path)
            if not os.path.exists(file_path):
                # If to terminate or not the session is lef to the user
                # other codes path may be present but not used
                logging.warning(f"Path {file_path} do not exist")
                # fatal_exception(file_path + ' does not exist')

        return file_path

    def read_settings(self) -> None:
        """
        Parse the configuration file

        Returns
        -------
        None.

        """

        conf_file = self.conf_file
        # Main
        main = pd.read_excel(
            conf_file, sheet_name="MAIN Config.", skiprows=1, header=None
        )
        main.columns = ["Variable", "Value"]
        main.set_index("Variable", inplace=True)
        self.mcnp_exec = self._process_path(main["Value"].loc["MCNP executable"])
        self.mcnp_config = self._process_path(main["Value"].loc["MCNP config"])
        self.serpent_exec = self._process_path(main["Value"].loc["Serpent executable"])
        self.serpent_config = self._process_path(main["Value"].loc["Serpent config"])
        self.openmc_exec = self._process_path(main["Value"].loc["OpenMC executable"])
        self.openmc_config = self._process_path(main["Value"].loc["OpenMC config"])
        self.d1s_exec = self._process_path(main["Value"].loc["d1S executable"])
        self.d1s_config = self._process_path(main["Value"].loc["d1S config"])
        self.openmp_threads = main["Value"].loc["OpenMP threads"]
        self.mpi_tasks = main["Value"].loc["MPI tasks"]
        self.mpi_exec_prefix = main["Value"].loc["MPI executable prefix"]
        self.batch_system = main["Value"].loc["Batch system"]
        self.batch_file = self._process_path(main["Value"].loc["Batch file"])

        """ Legacy config variables """
        # self.xsdir_path = main['Value'].loc['xsdir Path']
        # self.suppressW = main['Value'].loc['Suppress warnings']
        # self.multi_threads = main['Value'].loc['multithread']
        # self.cpu = main['Value'].loc['CPU']

        # Computational
        comp_default = pd.read_excel(
            conf_file, sheet_name="Computational benchmarks", skiprows=2
        )
        self.comp_default = comp_default.dropna(subset=["Folder Name"])

        # Experimental
        comp_default = pd.read_excel(
            conf_file, sheet_name="Experimental benchmarks", skiprows=2
        )
        self.exp_default = comp_default.dropna(subset=["Folder Name"])

        # Libraries
        lib = pd.read_excel(conf_file, sheet_name="Libraries", keep_default_na=False)
        self.lib = lib

        # self.default_lib = lib[lib['Default'] == 'yes']['Suffix'].values[0]

    def run_option(self, exp=False) -> str:
        """Present option of running in command line or submit as a job.

        Parameters
        ----------
        exp : bool, optional
            Whether an experimental benchmark, by default False

        Returns
        -------
        str
            c or s user selected option.
        """
        if exp:
            config = self.exp_default.set_index("Description")
        else:
            config = self.comp_default.set_index("Description")

        runoption = "c"

        if not sys.platform.startswith("win"):
            for testname, row in config.iterrows():
                if bool(row["OnlyInput"]):
                    runoption = "c"
                    break
            else:
                while True:
                    runoption = input(
                        " Would you like to run in the command line, c, or submit as a job, s? "
                    )
                    if runoption == "c":
                        break
                    elif runoption == "s":
                        if pd.isnull(self.batch_system):
                            print(
                                " Cannot submit as a batch job, as no batch system has been defined in the config file."
                            )
                        elif (pd.isnull(self.mpi_exec_prefix)) and (
                            pd.isnull(self.mpi_tasks) is not True
                        ):
                            if int(self.mpi_tasks) > 1:
                                print(
                                    " Cannot submit batch job as MPI, as no MPI executable prefix has been defined in the config file."
                                )
                        else:
                            break
                    elif runoption == "back":
                        break
                    elif runoption == "exit":
                        break
                    else:
                        print(" Please enter a valid option")

        return runoption

    def get_lib_name(self, suffix: str) -> str:
        """
        Get the name of the library from its suffix. If a name was not
        specified in the configuration file the same suffix is returned

        Parameters
        ----------
        suffix : str
            e.g. 21c .

        Returns
        -------
        str
            Name of the library.

        """
        df = self.lib.set_index("Suffix")
        # strip the suffix from dots, just in case
        suffix = suffix.strip(".")
        try:
            name = df.loc[suffix, "Name"]
        except KeyError:
            name = suffix
        return name


@dataclass
class BenchmarkConfiguration:
    """Configuration for a computational benchmark.

    Attributes
    ----------
    excel_options : dict[int, ExcelOptions]
        Options for the Excel benchmark.
    atlas_options : dict[int, AtlasOptions]
        Options for the Atlas benchmark.
    """

    excel_options: dict[int, ExcelOptions]
    atlas_options: dict[int, AtlasOptions]

    @classmethod
    def from_yaml(cls, file: str | os.PathLike) -> BenchmarkConfiguration:
        """Build the configuration for a computational benchmark from a yaml file.

        Parameters
        ----------
        file : str | os.PathLike
            path to the yaml file.

        Returns
        -------
        ComputationalConfig
            The configuration for the computational benchmark.
        """
        with open(file) as f:
            cfg = yaml.safe_load(f)

        atlas_options = {}
        for key, value in cfg["Atlas"].items():
            atlas_options[int(key)] = AtlasOptions(**value)

        excel_options = {}
        for key, value in cfg["Excel"].items():
            excel_options[int(key)] = ExcelOptions(**value)

        return cls(excel_options=excel_options, atlas_options=atlas_options)


@dataclass
class ExcelOptions:
    """Dataclass storing options for the Excel benchmark.

    Attributes
    ----------
    identifier : int
        Identifier of the tally.
    x : BinningType | list[BinningType]
        Tally dataframe column name to use for x axis. If a list, two binnings are
        combined together to form a single binning. The only valid combination is
        Cells-Segments for the moment.
    x_name : str
        X axis label.
    y : BinningType | list[BinningType]
        Tally dataframe column name to use for y axis. If a list, two binnings are
        combined together to form a single binning. The only valid combination is
        Cells-Segments for the moment.
    y_name : str
        Y axis label.

    Raises
    ------
    ValueError
        If an invalid BinningType is provided.
    """

    identifier: int  # identifier of the tally
    x: BinningType | list[BinningType]  # tally dataframe column name to use for x axis
    x_name: str  # x label
    y: BinningType | list[BinningType]  # tally dataframe column name to use for y axis
    y_name: str  # y label
    cut_y: int | None = (
        None  # max number of columns, after that the DF is split and goes to next line
    )
    compute_CE: bool = False  # compute the CE of the tally if requested

    def __post_init__(self):
        # enforce that the binning type is a valid one, try to convert if possible
        for attribute in [self.x, self.y]:
            if type(attribute) is str:
                split = attribute.split("-")
                if len(split) > 1:
                    attribute = []
                    for bintype in split:
                        try:
                            attribute.append(BinningType(bintype))
                        except ValueError:
                            raise ValueError(f"Invalid binning type: {bintype}")
                else:
                    try:
                        attribute = BinningType(attribute)
                    except ValueError:
                        raise ValueError(f"Invalid binning type: {attribute}")
        # try:
        #     self.x = BinningType(self.x)
        # except ValueError:
        #     raise ValueError(f"Invalid binning type for x: {self.x}")
        # try:
        #     self.y = BinningType(self.y)
        # except ValueError:
        #     raise ValueError(f"Invalid binning type for y: {self.y}")


@dataclass
class AtlasOptions:
    """
    AtlasOptions is a configuration data class for specifying options related to plotting
    in the Atlas application.

    Attributes
    ----------
    identifier : int
        Identifier of the tally.
    plot_type : PlotType
        Type of plot.
    quantity : str, optional
        Quantity plotted (goes on the y axis of plots). Default is None.
    unit : str, optional
        Unit of the quantity. Default is None.

    Raises
    ------
    ValueError
        if an invalid PlotType is provided.
    """

    identifier: int  # identifier of the tally
    plot_type: PlotType  # type of plot
    quantity: str | None = None  # quantity plotted (goes on the y axis of plots)
    unit: str | None = None  # unit of the quantity

    def __post_init__(self):
        # enforce that the plot type is a valid one
        try:
            self.plot_type = PlotType(self.plot_type)
        except ValueError:
            raise ValueError(f"Invalid plot type: {self.plot_type}")
