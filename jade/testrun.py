from __future__ import annotations

# -*- coding: utf-8 -*-

# Created on Mon Nov  4 16:52:09 2019

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


import os
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
import logging
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

import f4enix.input.MCNPinput as ipt
import jade.inputfile as inputfile
import f4enix.input.materials as mat
import jade.unix as unix
from jade.configuration import Configuration
from f4enix.input.libmanager import LibManager
from f4enix.input.d1suned import IrradiationFile, Reaction, ReactionFile
from jade.__version__ import __version__
from jade.__openmc__ import OMC_AVAIL

if OMC_AVAIL:
    import jade.openmc as omc


# colors
CRED = "\033[91m"
CORANGE = "\033[93m"
CEND = "\033[0m"


class Test:
    def __init__(
        self,
        inp: os.PathLike,
        lib: str,
        config: pd.DataFrame,
        confpath: os.PathLike,
        runoption: str,
        lib_name: str,
    ) -> None:
        """
        Class representing a general test. This class will have to be extended
        for specific tests.

        Parameters
        ----------
        inp : str
            path to inputfile blueprint.
        lib : str
            library suffix to use (e.g. 31c).
        config : pd.DataFrame (single row)
            configuration options for the test.
        confpath : path like object
            path to the test configuration folder.
        runoption : str
            flag for parallel execution. if 'c' is run in command line, if 's'
            is submitted as a job.
        lib_name : str
            extended name of the library.

        Raises
        ------
        ValueError
            if the code specified in config is not admissible.

        Returns
        -------
        None.

        """
        # Test Library
        self.lib = lib
        self.lib_name = lib_name

        # Parallel execution
        self.runoption = runoption

        # Configuration options for the test
        self.config = config

        # MCNP original input
        self.original_inp = inp

        # VRT path
        self.path_VRT = inp

        # Get the configuration files path
        self.test_conf_path = confpath

        # Input variables
        self.mcnp_ipt = None
        self.serpent_ipt = None
        self.openmc_ipt = None
        self.d1s_ipt = None

        self.irrad = None
        self.react = None

        # Path variables
        self.run_dir = None

        config = config.dropna()

        # self.name = config["Folder Name"]

        try:
            self.nps = config["NPS cut-off"]
        except KeyError:
            self.nps = None
        if self.nps is np.nan:
            self.nps = None

        # Updated to handle multiple codes
        try:
            self.mcnp = bool(config["MCNP"])
        except KeyError:
            self.mcnp = False
        try:
            self.serpent = bool(config["Serpent"])
        except KeyError:
            self.serpent = False
        try:
            self.openmc = bool(config["OpenMC"])
        except KeyError:
            self.openmc = False
        try:
            self.d1s = bool(config["d1S"])
        except KeyError:
            self.d1s = False

        # Generate input file template according to transport code
        if self.d1s:
            d1s_ipt = os.path.join(inp, "d1s", os.path.basename(inp) + ".i")
            irrfile = os.path.join(inp, "d1s", os.path.basename(inp) + "_irrad")
            reacfile = os.path.join(inp, "d1s", os.path.basename(inp) + "_react")
            self.d1s_inp = ipt.D1S_Input.from_input(d1s_ipt)
            try:
                self.irrad = IrradiationFile.from_text(irrfile)
                self.react = ReactionFile.from_text(reacfile)
                self.d1s_inp.irrad_file = self.irrad
                self.d1s_inp.reac_file = self.react
            except FileNotFoundError:
                logging.info("d1S irradition and reaction files not found, skipping...")
            self.name = os.path.basename(d1s_ipt).split(".")[0]
        if self.mcnp:
            mcnp_ipt = os.path.join(inp, "mcnp", os.path.basename(inp) + ".i")
            self.mcnp_inp = ipt.Input.from_input(mcnp_ipt)
            self.name = os.path.basename(mcnp_ipt).split(".")[0]
        if self.serpent:
            serpent_ipt = os.path.join(inp, "serpent", os.path.basename(inp) + ".i")
            self.serpent_inp = inputfile.SerpentInputFile.from_text(serpent_ipt)
        if self.openmc:
            openmc_ipt = os.path.join(inp, "openmc")
            self.openmc_inp = omc.OpenMCInputFiles(openmc_ipt)

    @staticmethod
    def _get_lib(lib: str | dict) -> str:
        """Get the library name.

        Parameters
        ----------
        lib : str | dict
            Library name.

        Returns
        -------
        str
            Library name.
        """
        if isinstance(lib, dict):
            lib = list(lib.values())[0]
        elif isinstance(lib, str):
            lib = lib
        return lib

    @staticmethod
    def _get_lib_d1s(lib: str | dict) -> str:
        """Get the library name.

        Parameters
        ----------
        lib : str | dict
            Library name.

        Returns
        -------
        str
            Library name.
        """

        # Handle 99c-31c format for SDDR benchmarks
        if isinstance(lib, dict):
            lib = list(lib.values())[0]
        elif isinstance(lib, str):
            if "-" in lib:
                lib = lib.split("-")[0]
            else:
                lib = lib
        return lib

    def _translate_input(self, lib, libmanager):
        """
        Translate the input template to selected library

        Parameters
        ----------
        lib : str or dic
            There are many ways to provide a librart to be translated
            check the matreader doc for more details.
        libmanager : libmanager.LibManager
            Manager dealing with libraries operations..

        Returns
        -------
        None.

        """

        if self.d1s:
            # Then it was the translation of a D1S input, additional
            # actions are required
            act_lib, tr_lib = check_transport_activation(lib)
            self.react = self.d1s_inp.get_reaction_file(
                libmanager, act_lib, set_as_attribute=True
            )
            self.d1s_inp.smart_translate(act_lib, tr_lib, libmanager)

            self.d1s_inp.update_zaidinfo(libmanager)
        if self.mcnp:
            self.mcnp_inp.translate(lib, libmanager)
            self.mcnp_inp.update_zaidinfo(libmanager)
        if self.serpent:
            # Add serpent file translation here
            pass
        if self.openmc:
            # Add openmc file translation here
            pass

    def generate_test(
        self,
        lib_directory: os.PathLike,
        libmanager: LibManager,
        run_dir: os.PathLike = None,
    ) -> None:
        """
        Generate the test input files

        Parameters
        ----------
        lib_directory : path or string
            Path to lib benchmarks input folders.
        libmanager : libmanager.LibManager
            Manager dealing with libraries operations.
        rundir : str or path
            allows to ovewrite the run directory if needed. The default is None

        Returns
        -------
        None.

        """
        # Translate the input
        self._translate_input(self.lib, libmanager)

        # Add stop card
        if self.d1s:
            self.d1s_inp.add_stopCard(self.nps)
        if self.mcnp:
            self.mcnp_inp.add_stopCard(self.nps)
        if self.serpent:
            self.serpent_inp.add_stopCard(self.nps)
        if self.openmc:
            self.openmc_inp.add_stopCard(self.nps)

        # Identify working directory
        # testname = self.inp.name
        testname = self.name
        if run_dir is None:
            motherdir = os.path.join(lib_directory, testname)
        else:
            motherdir = run_dir
        self.run_dir = motherdir
        # If previous results are present they are cancelled
        if os.path.exists(motherdir):
            shutil.rmtree(motherdir)
        os.mkdir(motherdir)

        # Allow space for personalization getting additional modification
        self.custom_inp_modifications()

        if self.d1s:
            # Create the ouput directory
            d1s_dir = os.path.join(motherdir, "d1s")
            os.mkdir(d1s_dir)
            outinpfile = os.path.join(d1s_dir, testname)
            self.d1s_inp.write(outinpfile)
            # And irradiation and reaction files if needed
            if self.irrad is not None:
                self.irrad.write(d1s_dir)
            if self.react is not None:
                self.react.write(d1s_dir)
            # Get WW files if available
            wwinp = os.path.join(self.original_inp, "d1s", "wwinp")
            if os.path.exists(wwinp):
                outfile = os.path.join(motherdir, "d1s", "wwinp")
                shutil.copyfile(wwinp, outfile)

        if self.mcnp:
            os.mkdir(os.path.join(motherdir, "mcnp"))
            outinpfile = os.path.join(motherdir, "mcnp", testname)
            self.mcnp_inp.write(outinpfile)
            # Get WW files if available
            wwinp = os.path.join(self.original_inp, "mcnp", "wwinp")
            if os.path.exists(wwinp):
                outfile = os.path.join(motherdir, "mcnp", "wwinp")
                shutil.copyfile(wwinp, outfile)

        if self.serpent:
            # Implement serpent outputfile generation here
            pass

        if self.openmc:
            # Implement openmc outputfile generation here
            pass

        # Print metadata
        self._print_metadata(motherdir)

    def _print_metadata(self, outpath: os.PathLike) -> None:
        """Print metadata file in the run directory. outpath is the path
        to the run directory excluding the code"""

        code_tags = self._get_code_tags()
        # read, update and add metadata file to the run directory
        # if it is a multiple run, there is a trick to check by the path
        pieces = self.original_inp.split(os.sep)
        if pieces[-2] in pieces[-1]:
            # then is a multiple run and we should go up one level
            metadata_inp = os.path.join(
                os.path.dirname(self.original_inp), "benchmark_metadata.json"
            )
        else:
            metadata_inp = os.path.join(self.original_inp, "benchmark_metadata.json")

        try:
            with open(metadata_inp, "r", encoding="utf-8") as f:
                metadata_inp = json.load(f)
        except FileNotFoundError:
            logging.warning(
                "Metadata file not found in %s", os.path.dirname(metadata_inp)
            )
            metadata_inp = {"name": self.name}

        for code_tag in code_tags:
            metadata = {}
            metadata["benchmark_name"] = metadata_inp["name"]
            try:
                metadata["benchmark_version"] = metadata_inp["version"][code_tag]
            except KeyError:
                metadata["benchmark_version"] = None
            metadata["jade_run_version"] = __version__
            metadata["library"] = self.lib_name
            metadata["code"] = code_tag
            outfile = os.path.join(outpath, code_tag, "metadata.json")
            try:
                with open(outfile, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=4)
            except FileNotFoundError:
                # It may happen that the code bool is set to true but the
                # corresponding directory is not created because generation is
                # not implemented yet. Simply do not print the metadata in this
                # case
                logging.warning('metadata "%s" cannot be created', outfile)

    def _get_code_tags(self) -> list[str]:
        codes = []
        if self.mcnp:
            codes.append("mcnp")
        if self.serpent:
            codes.append("serpent")
        if self.openmc:
            codes.append("openmc")
        if self.d1s:
            codes.append("d1s")
        return codes

    def custom_inp_modifications(self):
        """
        Perform additional operation on the input before generation. In this
        parent object actually does nothing

        Returns
        -------
        None.

        """
        # It does not do anything in the default benchmark
        pass

    def run(self, config, libmanager, runoption: str) -> None:
        """
        run the input

        Parameters
        ----------
        config :
            Configuration settings
        libmanager :
            libmanager
        runoption : str
        """

        directory = self.run_dir
        name = self.name
        if self.d1s:
            lib = self._get_lib_d1s(self.lib)
        else:
            lib = self._get_lib(self.lib)

        if self.d1s:
            d1s_directory = os.path.join(directory, "d1s")
            self.run_mcnp(
                lib, config, libmanager, name, d1s_directory, runoption, d1s=True
            )

        if self.mcnp:
            mcnp_directory = os.path.join(directory, "mcnp")
            self.run_mcnp(lib, config, libmanager, name, mcnp_directory, runoption)

        if self.serpent:
            serpent_directory = os.path.join(directory, "serpent")
            self.run_serpent(
                lib, config, libmanager, name, serpent_directory, runoption
            )

        if self.openmc:
            openmc_directory = os.path.join(directory, "openmc")
            self.run_openmc(lib, config, libmanager, openmc_directory, runoption)

    # Edited by D.Wheeler, UKAEA
    # Job submission currently tailored for LoadLeveler, may be applicable to other submission systems with equivalent dummy variables
    @staticmethod
    def job_submission(
        config,
        directory: str,
        run_command: list,
        mpi_tasks: int,
        omp_threads: int,
        env_variables: str,
        data_command=str(),
    ) -> None:
        """Submits a job script to the users batch system for running in parallel.

        Parameters
        ----------
        config :
            Configuration settings
        directory :
            Job working directory
        run_command : list
            Executable command
        mpi_tasks : int
            Number of MPI tasks
        omp_threads : int
            Number of OMP threads
        env_variables : str
            user specified/ code specific environment variables
        data_command : str, optional
            user specified/ code specific environment variables, by default str()
        """

        # Read contents of batch file template.
        with open(env_variables, "r") as f:
            config_script = ""
            for line in f:
                if not line.startswith("#!"):
                    config_script += line
        user = (
            subprocess.run("whoami", capture_output=True).stdout.decode("utf-8").strip()
        )
        os.chdir(directory)
        job_script = os.path.join(
            directory, os.path.basename(directory) + "_job_script"
        )
        essential_commands = ["MPI_TASKS"]
        with open(config.batch_file, "rt") as fin, open(job_script, "wt") as fout:
            # Replace placeholders in batch file template with actual values
            contents = fin.read()
            for cmd in essential_commands:
                if cmd not in contents:
                    raise Exception(
                        "Unable to find essential dummy variable {} in job "
                        "script template, please check and re-run".format(cmd)
                    )
            contents = contents.replace("INITIAL_DIR", directory)
            contents = contents.replace("OUT_FILE", job_script + ".out")
            contents = contents.replace("ERROR_FILE", job_script + ".err")
            contents = contents.replace("MPI_TASKS", str(mpi_tasks))
            contents = contents.replace("OMP_THREADS", str(omp_threads))
            contents = contents.replace("USER", user)

            contents += "\n\n" + config_script
            contents += "\n\n" + str(data_command)
            contents += "\n\n" + " ".join(run_command)

            fout.write(contents)

        # Submit the job using the specified batch system
        subprocess.run(
            config.batch_system + " " + job_script, cwd=directory, shell=True
        )

    @staticmethod
    def run_mcnp(
        lib: str,
        config,
        lib_manager,
        name: str,
        directory: Path,
        runoption: str,
        timeout=None,
        d1s=False,
    ) -> bool:
        """Run MCNP simulation either on the command line or submitted as a job.

        Parameters
        ----------
        lib : str
            library to be run, needed to get the correct xsdir file
        config :
            Configuration settings
        lib_manager :
            libmanager
        name : str
            Name of the simulation
        directory : str, path
            Directory where the simulation will be executed
        runoption: str
            Whether JADE run in parallel or command line
        timeout : float, optional
            Maximum time to wait for simulation of complete, by default None
        d1s : bool, optional
            Flag to run d1s, by default False

        Returns
        -------
        bool
            Flag if simulation not run
        """

        # Calculate MPI tasks and OpenMP threads
        mpi_tasks = int(config.mpi_tasks)
        omp_threads = int(config.openmp_threads)
        run_mpi = False
        if mpi_tasks > 1:
            run_mpi = True
        run_openmp = False
        if omp_threads > 1:
            run_openmp = True

        executable = config.mcnp_exec
        env_variables = config.mcnp_config
        inputstring = "i=" + name
        outputstring = "n=" + name
        tasks = "tasks " + str(omp_threads)
        libpath = Path(str(lib_manager.data["mcnp"][lib].filename))
        data_command = "export DATAPATH=" + str(libpath.parent)

        if d1s == True:
            xsstring = "xs=" + str(lib_manager.data["d1s"][lib].filename)
            executable = config.d1s_exec
            env_variables = config.d1s_config
        else:
            xsstring = "xs=" + str(lib_manager.data["mcnp"][lib].filename)
            executable = config.mcnp_exec
            env_variables = config.mcnp_config

        flagnotrun = False

        if pd.isnull(executable) is not True:
            run_command = [executable, inputstring, outputstring, xsstring]
            if run_openmp:
                run_command.append(tasks)

            try:
                cwd = os.getcwd()
                os.chdir(directory)
                # cancel eventual previous output file
                outputfile = name + ".o"
                if os.path.exists(outputfile):
                    os.remove(outputfile)

                # check if runtpe exists
                runtpe = name + ".r"
                if os.path.exists(runtpe):
                    command = command + " runtpe=" + name + ".r"

                if runoption.lower() == "c":
                    try:
                        # in case of run in console dump the prints to a file
                        # to get the code version in the metadata
                        run_command.append(" > dump.out")
                        os.environ["DATAPATH"] = str(libpath.parent)
                        if not sys.platform.startswith("win"):
                            unix.configure(env_variables)
                        print(" ".join(run_command))
                        subprocess.run(
                            " ".join(run_command),
                            cwd=directory,
                            shell=True,
                            timeout=43200,
                        )

                    except subprocess.TimeoutExpired:
                        print(
                            "Sesion timed out after 12 hours. Consider submitting as a job."
                        )
                        flagnotrun = True

                elif runoption.lower() == "s":
                    if run_mpi:
                        run_command.insert(0, config.mpi_exec_prefix)
                    # Run MCNP as a job
                    cwd = os.getcwd()
                    os.chdir(directory)
                    Test.job_submission(
                        config,
                        directory,
                        run_command,
                        mpi_tasks,
                        omp_threads,
                        env_variables,
                        data_command,
                    )
                    os.chdir(cwd)
            except subprocess.TimeoutExpired:
                pass

        return flagnotrun

    @staticmethod
    def run_serpent(
        lib: str,
        config: Configuration,
        lib_manager: LibManager,
        name: str,
        directory: Path,
        runoption: str,
    ) -> bool:
        """Run Serpent simulation either on the command line or submitted as a job.

        Parameters
        ----------
        lib : str
            library to assess. needed to recover path to nuclear data
        config :
            Configuration settings
        lib_manager :
            libmanager
        name : str
            Name of the simulation
        directory : str, path
            Directory where the simulation will be executed
        runoption: str
            Whether JADE run in parallel or command line
        timeout : float, optional
            Maximum time to wait for simulation of complete, by default None

        Returns
        -------
        bool
            Flag if simulation not run
        """

        # Calculate MPI tasks and OpenMP threads
        mpi_tasks = int(config.mpi_tasks)
        omp_threads = int(config.openmp_threads)
        run_mpi = False
        if mpi_tasks > 1:
            run_mpi = True
        run_openmp = False
        if omp_threads > 1:
            run_openmp = True

        executable = config.serpent_exec
        env_variables = config.serpent_config
        inputstring = name
        libpath = Path(str(lib_manager.data["serpent"][lib].filename))
        data_command = (
            "export SERPENT_DATA="
            + str(libpath.parent)
            + " \nexport SERPENT_ACELIB="
            + str(libpath)
        )

        flagnotrun = False

        if pd.isnull(executable) is not True:
            run_command = [executable, inputstring]
            if run_openmp:
                run_command = [executable, "-omp", str(omp_threads), inputstring]

            if runoption.lower() == "c":
                try:
                    os.environ["SERPENT_DATA"] = str(libpath.parent)
                    os.environ["SERPENT_ACELIB"] = str(str(libpath))
                    unix.configure(env_variables)
                    print(" ".join(run_command))
                    subprocess.run(
                        " ".join(run_command), cwd=directory, shell=True, timeout=43200
                    )

                except subprocess.TimeoutExpired:
                    print(
                        "Sesion timed out after 12 hours. Consider submitting as a job."
                    )
                    flagnotrun = True

            elif runoption.lower() == "s":
                if run_mpi:
                    run_command.insert(0, config.mpi_exec_prefix)
                # Run Serpent as a job
                cwd = os.getcwd()
                os.chdir(directory)
                Test.job_submission(
                    config,
                    directory,
                    run_command,
                    mpi_tasks,
                    omp_threads,
                    env_variables,
                    data_command,
                )
                os.chdir(cwd)

        return flagnotrun

    @staticmethod
    def run_openmc(
        lib: str,
        config: Configuration,
        lib_manager: LibManager,
        directory: os.PathLike,
        runoption: str,
    ) -> bool:
        """Run OpenMC simulation either on the command line or submitted as a job.

        Parameters
        ----------
        lib: str
            library to assess. needed to recover path to nuclear data
        config :
            Configuration settings
        lib_manager :
            libmanager
        directory : str, path
            Directory where the simulation will be executed
        runoption: str
            Whether JADE run in parallel or command line

        Returns
        -------
        bool
            Flag if simulation not run
        """

        # Calculate MPI tasks and OpenMP threads
        mpi_tasks = int(config.mpi_tasks)
        omp_threads = int(config.openmp_threads)
        run_mpi = False
        if mpi_tasks > 1:
            run_mpi = True
        run_openmp = False
        if omp_threads > 1:
            run_openmp = True

        executable = config.openmc_exec
        env_variables = config.openmc_config
        libpath = Path(str(lib_manager.data["openmc"][lib].filename))
        data_command = "export OPENMC_CROSS_SECTIONS=" + str(libpath)

        flagnotrun = False

        if pd.isnull(executable) is not True:
            run_command = [executable]
            # Run OpenMC from command line either OMP, MPI or hybrid MPI-OMP
            if run_openmp:
                run_command = [executable, "--threads", str(omp_threads)]

            if runoption.lower() == "c":
                try:
                    os.environ["OPENMC_CROSS_SECTIONS"] = str(libpath)
                    unix.configure(env_variables)
                    print(" ".join(run_command))
                    subprocess.run(
                        " ".join(run_command), cwd=directory, shell=True, timeout=43200
                    )

                except subprocess.TimeoutExpired:
                    print(
                        "Sesion timed out after 12 hours. Consider submitting as a job."
                    )
                    flagnotrun = True

            elif runoption.lower() == "s":
                if run_mpi:
                    run_command.insert(0, config.mpi_exec_prefix)
                # Run OpenMC as a job
                cwd = os.getcwd()
                os.chdir(directory)
                Test.job_submission(
                    config,
                    directory,
                    run_command,
                    mpi_tasks,
                    omp_threads,
                    env_variables,
                    data_command,
                )
                os.chdir(cwd)

        return flagnotrun


class SphereTest(Test):
    """
    Class handling the sphere test
    """

    def generate_test(self, directory, libmanager, limit=None, lib=None):
        """
        Generated all the sphere test for a selected library

        Parameters
        ----------
        directory : str or path
            path to the sphere input folder.
        libmanager : LibManager
            manager of the nuclear data operations.
        limit : int, optional
            limit the test to the first n zaids and materials.
            The default is None.

        Returns
        -------
        None.

        """
        if lib is None:
            lib = self.lib
        # Get typical materials input
        dirmat = os.path.dirname(self.original_inp)
        matpath = os.path.join(dirmat, "TypicalMaterials")
        inpmat = ipt.Input.from_input(matpath)
        materials = inpmat.materials

        # Get zaids available in the selected library
        if self.d1s:
            testname = "SphereSDDR"
        else:
            testname = "Sphere"
        zaids = libmanager.get_libzaids(lib, "mcnp")

        motherdir = os.path.join(directory, testname)
        # If previous results are present they are deleted
        if os.path.exists(motherdir):
            shutil.rmtree(motherdir)
        os.mkdir(motherdir)

        # GET SETTINGS
        # Zaids
        settings = os.path.join(self.test_conf_path, "ZaidSettings.csv")
        settings = pd.read_csv(settings, sep=",").set_index("Z")
        # Materials
        settings_mat = os.path.join(self.test_conf_path, "MaterialsSettings.csv")
        settings_mat = pd.read_csv(settings_mat, sep=",").set_index("Symbol")

        self.run_dir = motherdir

        print(" Zaids:")
        # for zaid in tqdm(zaids):
        for zaid in tqdm(zaids[:limit]):
            Z = int(zaid[:-3])
            # Get Density
            density = settings.loc[Z, "Density [g/cc]"]

            if settings.loc[Z, "Let Override"]:
                # get stop parameters
                if self.nps is None:
                    nps = settings.loc[Z, "NPS cut-off"]
                    if nps is np.nan:
                        nps = None
                else:
                    nps = self.nps

            # Zaid local settings are prioritized
            else:
                nps = settings.loc[Z, "NPS cut-off"]
                if nps is np.nan:
                    nps = None

            self.generate_zaid_test(
                zaid, libmanager, testname, motherdir, -1 * density, nps
            )

        print(" Materials:")
        # for material in tqdm(materials.materials):
        for material in tqdm(materials.materials[:limit]):
            # Get density
            density = settings_mat.loc[material.name.upper(), "Density [g/cc]"]

            self.generate_material_test(
                material, -1 * density, libmanager, testname, motherdir
            )

    def generate_zaid_test(
        self,
        zaid,
        libmanager,
        testname,
        motherdir,
        density,
        nps,
        addtag=None,
        lib=None,
    ):
        """
        Generate input for a single zaid sphere leakage benchmark run.

        Parameters
        ----------
        zaid : str
            zaid in string format.
        libmanager : Libmanager
            Jade Libmanager.
        testname : str
            name of the benchmark.
        motherdir : str/path
            Path to the benchmark folder.
        density : (str/float)
            Density value for the sphere.
        nps : float
            number of particles cut-off
        addtag : str, optional
            add tag at the end of the single zaid test name. The default is
            None
        parentlist : list, optional
            add the PIKMT if requested (list of parent zaids)

        Returns
        -------
        None.

        """
        if lib is None:
            lib = self.lib

        # Adjourn the material cards for the zaid
        zaid = mat.Zaid(1, zaid[:-3], zaid[-3:], lib)
        name, formula = libmanager.get_zaidname(zaid)

        if self.d1s:
            # Retrieve wwinp & other misc files if they exist
            directoryVRT = os.path.join(
                self.path_VRT, "d1s", zaid.element + zaid.isotope
            )
            edits_file = os.path.join(directoryVRT, "inp_edits.txt")
            ww_file = os.path.join(directoryVRT, "wwinp")
            # Create MCNP material card
            submat = mat.SubMaterial("M1", [zaid], header="C " + name + " " + formula)
            material = mat.Material([zaid], None, "M1", submaterials=[submat])
            materials = mat.MatCardsList([material])

            # Generate the new input
            newinp = deepcopy(self.d1s_inp)
            newinp.materials = materials  # Assign material
            # adjourn density
            sphere_cell = newinp.cells["2"]
            sphere_cell.set_d(str(density))
            sphere_cell.lines = sphere_cell.card()
            # assign stop card
            newinp.add_stopCard(nps)
            # add PIKMT if requested
            newinp.add_PIKMT_card()

            # Write new input file
            outfile, outdir = self._get_zaidtestname(
                testname, zaid, formula, addtag=addtag
            )

            outpath = os.path.join(motherdir, outdir, "d1s")
            os.makedirs(outpath, exist_ok=True)
            outinpfile = os.path.join(outpath, outfile)
            newinp.write(outinpfile)
            newinp.irrad_file.write(outpath)
            newinp.reac_file.write(outpath)

            # Copy also wwinp file
            if os.path.exists(directoryVRT):
                outwwfile = os.path.join(outpath, "wwinp")
                shutil.copyfile(ww_file, outwwfile)

        if self.mcnp:
            # Retrieve wwinp & other misc files if they exist
            directoryVRT = os.path.join(
                self.path_VRT, "mcnp", zaid.element + zaid.isotope
            )
            edits_file = os.path.join(directoryVRT, "inp_edits.txt")
            ww_file = os.path.join(directoryVRT, "wwinp")
            # Create MCNP material card
            submat = mat.SubMaterial("M1", [zaid], header="C " + name + " " + formula)
            material = mat.Material([zaid], None, "M1", submaterials=[submat])
            materials = mat.MatCardsList([material])

            # Generate the new input
            newinp = deepcopy(self.mcnp_inp)
            newinp.materials = materials  # Assign material
            # adjourn density
            sphere_cell = newinp.cells["2"]
            sphere_cell.set_d(str(density))
            sphere_cell.lines = sphere_cell.card()
            # assign stop card
            newinp.add_stopCard(nps)
            # Write new input file
            outfile, outdir = self._get_zaidtestname(
                testname, zaid, formula, addtag=addtag
            )

            outpath = os.path.join(motherdir, outdir, "mcnp")
            os.makedirs(outpath, exist_ok=True)
            outinpfile = os.path.join(outpath, outfile)
            newinp.write(outinpfile)

            # Copy also wwinp file
            if os.path.exists(directoryVRT):
                outwwfile = os.path.join(outpath, "wwinp")
                shutil.copyfile(ww_file, outwwfile)

        if self.serpent:
            # Create Serpent material card
            submat = mat.SubMaterial(
                "mat 1", [zaid], header="% " + name + " " + formula
            )
            material = mat.Material(
                [zaid], None, "mat 1", submaterials=[submat], density=density
            )
            materials = mat.MatCardsList([material])

            # Generate the new input
            newinp = deepcopy(self.serpent_inp)
            newinp.materials = materials  # Assign material

            # assign stop card
            newinp.add_stopCard(nps)

            # Write new input file
            outfile, outdir = self._get_zaidtestname(
                testname, zaid, formula, addtag=addtag
            )
            outpath = os.path.join(motherdir, outdir, "serpent")
            os.makedirs(outpath, exist_ok=True)
            outinpfile = os.path.join(outpath, outfile)
            newinp.write(outinpfile)

        if self.openmc:
            # Create OpenMC material card
            submat = mat.SubMaterial("m1", [zaid])
            material = mat.Material(
                [zaid], None, "m1", submaterials=[submat], density=density
            )
            materials = mat.MatCardsList([material])

            # Generate the new input
            newinp = deepcopy(self.openmc_inp)

            # Assign material
            newinp.matlist_to_openmc(materials, libmanager)

            # assign stop card
            newinp.add_stopCard(nps)

            # Write new input file
            outfile, outdir = self._get_zaidtestname(
                testname, zaid, formula, addtag=addtag
            )
            outpath = os.path.join(motherdir, outdir, "openmc")
            os.makedirs(outpath, exist_ok=True)
            newinp.write(outpath)

        self._print_metadata(os.path.join(motherdir, outdir))

    @staticmethod
    def _get_zaidtestname(testname, zaid, formula, addtag=None):
        outfile = testname + "_" + zaid.element + zaid.isotope + "_" + formula + "_"
        outdir = testname + "_" + zaid.element + zaid.isotope + "_" + formula

        if addtag is not None:
            outfile = outfile + addtag + "_"
            outdir = outdir + "_" + addtag

        return outfile, outdir

    def generate_material_test(
        self,
        material,
        density,
        libmanager,
        testname,
        motherdir,
        lib=None,
    ):
        """
        Generate a sphere leakage benchmark input for a single typical
        material.

        Parameters
        ----------
        material : matreader.Material
            material object to be used for the new input.
        density : float
            densitiy value in g/cc
        libmanager : Libmanager
            Jade Libmanager.
        testname : str
            name of the benchmark.
        motherdir : str/path
            Path to the benchmark folder.

        Returns
        -------
        None.

        """
        if lib is None:
            lib = self.lib
        truename = material.name

        if self.d1s:
            # Retrieve wwinp & other misc files if they exist
            directoryVRT = os.path.join(self.path_VRT, "d1s", truename)
            edits_file = os.path.join(directoryVRT, "inp_edits.txt")
            ww_file = os.path.join(directoryVRT, "wwinp")
            newmat = deepcopy(material)

            # Translate and assign the material
            newmat.translate(lib, libmanager, "d1s")
            newmat.header = material.header + "C\nC True name:" + truename
            newmat.name = "M1"
            materials = mat.MatCardsList([newmat])

            # Generate the new input
            newinp = deepcopy(self.d1s_inp)
            newinp.materials = materials  # Assign material
            # adjourn density
            sphere_cell = newinp.cells["2"]
            sphere_cell.set_d(str(density))
            sphere_cell.lines = sphere_cell.card()
            # add stop card
            newinp.add_stopCard(self.nps)
            # --- Add the reaction file ---
            # python dicts are ordered now, first entry is activation lib
            activation_lib = list(lib.keys())[0]
            newinp.get_reaction_file(libmanager, activation_lib)
            # Add PIKMT card if required
            newinp.add_PIKMT_card()
            # Add the tracking for the contributions of different parents
            # and daughthers
            # delete the default FU card and assign parent tracking
            del newinp.other_data["FU104"]
            newinp.add_track_contribution(
                "F104", newinp.reac_file.get_parents(), who="parent"
            )

            # Write new input file
            outfile = testname + "_" + truename + "_"
            outdir = testname + "_" + truename

            outpath = os.path.join(motherdir, outdir, "d1s")
            os.makedirs(outpath, exist_ok=True)
            outinpfile = os.path.join(outpath, outfile)
            newinp.write(outinpfile)
            newinp.irrad_file.write(outpath)
            newinp.reac_file.write(outpath)

            # Copy also wwinp file
            if os.path.exists(directoryVRT):
                outwwfile = os.path.join(outpath, "wwinp")
                shutil.copyfile(ww_file, outwwfile)

        if self.mcnp:
            # Retrieve wwinp & other misc files if they exist
            directoryVRT = os.path.join(self.path_VRT, "mcnp", truename)
            edits_file = os.path.join(directoryVRT, "inp_edits.txt")
            ww_file = os.path.join(directoryVRT, "wwinp")
            newmat = deepcopy(material)

            # Translate and assign the material
            newmat.translate(lib, libmanager, "mcnp")
            newmat.header = material.header + "C\nC True name:" + truename
            newmat.name = "M1"
            materials = mat.MatCardsList([newmat])

            # Generate the new input
            newinp = deepcopy(self.mcnp_inp)
            newinp.materials = materials  # Assign material
            # adjourn density
            sphere_cell = newinp.cells["2"]
            sphere_cell.set_d(str(density))
            sphere_cell.lines = sphere_cell.card()
            # add stop card
            newinp.add_stopCard(self.nps)

            # Write new input file
            outfile = testname + "_" + truename + "_"
            outdir = testname + "_" + truename

            outpath = os.path.join(motherdir, outdir, "mcnp")
            os.makedirs(outpath, exist_ok=True)
            outinpfile = os.path.join(outpath, outfile)
            newinp.write(outinpfile)

            # Copy also wwinp file
            if os.path.exists(directoryVRT):
                outwwfile = os.path.join(outpath, "wwinp")
                shutil.copyfile(ww_file, outwwfile)

        if self.serpent:
            newmat = deepcopy(material)
            # Translate and assign the material
            newmat.translate(lib, libmanager, "serpent")
            newmat.header = material.header + "%\n% True name:" + truename
            newmat.name = "mat 1"
            newmat.density = density
            materials = mat.MatCardsList([newmat])

            # Generate the new input
            newinp = deepcopy(self.serpent_inp)
            newinp.materials = materials  # Assign material
            # add stop card
            newinp.add_stopCard(self.nps)

            # Write new input file
            outfile = testname + "_" + truename + "_"
            outdir = testname + "_" + truename

            outpath = os.path.join(motherdir, outdir, "serpent")
            os.makedirs(outpath, exist_ok=True)
            outinpfile = os.path.join(outpath, outfile)
            newinp.write(outinpfile)

        if self.openmc:
            newmat = deepcopy(material)
            newmat.name = "m1"
            newmat.density = density
            materials = mat.MatCardsList([newmat])

            # Generate the new input
            newinp = deepcopy(self.openmc_inp)

            # Assign material
            newinp.matlist_to_openmc(materials, libmanager)

            # add stop card
            newinp.add_stopCard(self.nps)

            # Write new input file
            outfile = testname + "_" + truename + "_"
            outdir = testname + "_" + truename

            outpath = os.path.join(motherdir, outdir, "openmc")
            os.makedirs(outpath, exist_ok=True)
            newinp.write(outpath)

        self._print_metadata(os.path.join(motherdir, outdir))

    def run(self, config, libmanager, runoption: str) -> None:
        """Sphere leakage requries ad-hoc run method.

        Parameters
        ----------
        config :
            Configuration settings
        libmanager :
            libmanager
        runoption : str
            Whether to run in the command line or submit as a job.
        """

        directory = self.run_dir
        if self.d1s:
            lib = self._get_lib_d1s(self.lib)
        else:
            lib = self._get_lib(self.lib)

        if self.d1s:
            d1s_directory = os.path.join(directory)
            for folder in tqdm(os.listdir(d1s_directory)):
                run_directory = os.path.join(d1s_directory, folder, "d1s")
                self.run_mcnp(
                    lib,
                    config,
                    libmanager,
                    folder + "_",
                    run_directory,
                    runoption,
                    d1s=True,
                )

        if self.mcnp:
            mcnp_directory = os.path.join(directory)
            for folder in tqdm(os.listdir(mcnp_directory)):
                run_directory = os.path.join(mcnp_directory, folder, "mcnp")
                self.run_mcnp(
                    lib, config, libmanager, folder + "_", run_directory, runoption
                )

        if self.serpent:
            serpent_directory = os.path.join(directory)
            for folder in tqdm(os.listdir(serpent_directory)):
                run_directory = os.path.join(serpent_directory, folder, "serpent")
                self.run_serpent(
                    lib, config, libmanager, folder + "_", run_directory, runoption
                )

        if self.openmc:
            openmc_directory = os.path.join(directory)
            for folder in tqdm(os.listdir(openmc_directory)):
                run_directory = os.path.join(openmc_directory, folder, "openmc")
                self.run_openmc(lib, config, libmanager, run_directory, runoption)


class SphereTestSDDR(SphereTest):
    def __init__(self, *args, **keyargs):
        super().__init__(*args, **keyargs)
        # Lib needs to be provided in the {activation lib}-{transportlib} format
        activationlib, transportlib = check_transport_activation(self.lib)
        self.activationlib = activationlib
        self.transportlib = transportlib

    def generate_test(self, directory, libmanager, limit=None, lib=None):
        super().generate_test(
            directory, libmanager, limit=limit, lib=self.activationlib
        )

    def generate_zaid_test(self, zaid, libmanager, testname, motherdir, density, nps):
        """
        Generate input for a single zaid sphere SDDR benchmark run.
        Depending on the number of reactions, multiple inputs may be generated
        from a single zaid.

        Parameters
        ----------
        zaid : str
            zaid in string format.
        libmanager : Libmanager
            Jade Libmanager.
        testname : str
            name of the benchmark.
        motherdir : str/path
            Path to the benchmark folder.
        density : (str/float)
            Density value for the sphere.
        nps : float
            number of particles cut-off

        Returns
        -------
        None.
        """

        # Recover the available reactions
        reactions = libmanager.get_reactions(self.activationlib, zaid)

        # Genearate a different test for each reaction
        for reaction in reactions:
            MT = reaction[0]
            daughter = reaction[1]
            try:
                filepath = os.path.join(
                    self.test_conf_path, "irrad_" + self.activationlib
                )
            except FileNotFoundError:
                raise FileNotFoundError("Irradiation file could not be found.")
            # --- Add the irradiation file ---
            self.d1s_inp.irrad_file = IrradiationFile.from_text(filepath)
            ans = self.d1s_inp.irrad_file.select_daughters_irradiation_file([daughter])
            # generate the reaction file with only the specific MT
            reaction = Reaction(f"{zaid}.{self.activationlib}", MT, daughter)
            reacfile = ReactionFile([reaction])
            self.d1s_inp.reac_file = reacfile

            # generate the input file
            super().generate_zaid_test(
                zaid,
                libmanager,
                testname,
                motherdir,
                density,
                nps,
                addtag=MT,
                lib=self.activationlib,
            )

            if not ans:
                print(
                    CORANGE
                    + " Warning: {}-{} irr file was not generated".format(zaid, MT)
                    + CEND
                )

    def generate_material_test(
        self, material, density, libmanager, testname, motherdir
    ):
        """
        Generate a sphere leakage benchmark input for a single typical
        material.

        Parameters
        ----------
        material : matreader.Material
            material object to be used for the new input.
        density : float
            densitiy value in g/cc
        libmanager : Libmanager
            Jade Libmanager.
        testname : str
            name of the benchmark.
        motherdir : str/path
            Path to the benchmark folder.

        Returns
        -------
        None.

        """
        # there will only be one test for each material that includes
        # all the possible reactions
        truename = material.name

        # --- Add the reaction file ---
        # Recover all the reactions (for each isotope) in the material
        reactions = []
        parentlist = []
        daughterlist = []
        transportlist = []
        for submat in material.submaterials:
            for zaid in submat.zaidList:
                parent = zaid.element + zaid.isotope
                zaidreactions = libmanager.get_reactions(self.activationlib, parent)
                if len(zaidreactions) > 0:
                    # it is a parent only if reactions are available
                    parentlist.append(parent)
                else:
                    # normal transport
                    transportlist.append(parent)

                for MT, daughter in zaidreactions:
                    reactions.append((parent, MT, daughter))
                    daughterlist.append(daughter)

        # eliminate duplicates
        daughterlist = list(set(daughterlist))
        parentlist = list(set(parentlist))
        transportlist = list(set(transportlist))

        # The generation of the inputs has to be done only if there is at
        # least one parent
        if len(parentlist) == 0:
            return
        else:
            # generate the input
            libs = {self.activationlib: parentlist, self.transportlib: transportlist}
            try:
                filepath = os.path.join(
                    self.test_conf_path, "irrad_" + self.activationlib
                )
            except FileNotFoundError:
                raise FileNotFoundError("Irradiation file could not be found.")

            # --- Add the irradiation file ---
            self.d1s_inp.irrad_file = IrradiationFile.from_text(filepath)
            ans = self.d1s_inp.irrad_file.select_daughters_irradiation_file(
                daughterlist
            )

            super().generate_material_test(
                material,
                density,
                libmanager,
                testname,
                motherdir,
                lib=libs,
            )

            # recover output directory and write file
            outdir = testname + "_" + truename

            if not ans:
                print(
                    CORANGE
                    + " Warning: {} irr file was not generated".format(outdir)
                    + CEND
                )


class FNGTest(Test):
    def custom_inp_modifications(self):
        # Add the tracking for daughters in tally 14
        zaids = self.irrad.get_daughters()
        self.d1s_inp.add_track_contribution("F14:p", zaids, who="daughter")
        # Add the tracking for daughters in tally 24
        zaids = self.react.get_parents()
        self.d1s_inp.add_track_contribution("F24:p", zaids, who="parent")


class MultipleTest:
    def __init__(
        self,
        inpsfolder: os.PathLike,
        lib: str,
        config: pd.DataFrame,
        confpath: os.PathLike,
        runoption: str,
        lib_name: str,
        TestOb=Test,
    ):
        """
        A collection of Tests

        Parameters
        ----------
        inpsfolder : path-like object
            folder that contains all inputs of the tests.
        lib : str
            library suffix to use (e.g. 31c).
        config : pd.DataFrame (single row)
            configuration options for the test.
        confpath : path like object
            path to the test configuration folder.
        lib_name : str
            name of the library to be used.
        TestOb : testrun.Test, optional
            type of test object to be used. The default is Test.

        Returns
        -------
        None.

        """
        tests = []
        for folder in os.listdir(inpsfolder):
            inp = os.path.join(inpsfolder, folder)
            if os.path.isfile(inp):
                continue
            test = TestOb(inp, lib, config, confpath, runoption, lib_name)
            tests.append(test)
        self.tests = tests
        self.name = os.path.basename(inpsfolder)

    def generate_test(self, lib_directory, libmanager):
        """
        Generate all the tests of the collection

        Parameters
        ----------
        lib_directory : path-like
            output directory where to generate the tests.
        libmanager : libmanager.LibManager
            object handling libraries operations.

        Returns
        -------
        None.

        """
        self.MCNPdir = os.path.join(lib_directory, self.name)
        safe_override(self.MCNPdir)
        for test in self.tests:
            mcnp_dir = os.path.join(self.MCNPdir, test.name)
            test.generate_test(lib_directory, libmanager, run_dir=mcnp_dir)

    def run(self, config, libmanager, runoption: str) -> None:
        """Run all tests

        Parameters
        ----------
        config :
            Configuration settings
        libmanager :
            libmanager
        runoption : str
            command line or as a job
        """
        for test in tqdm(self.tests):
            test.run(config, libmanager, runoption)


def safe_mkdir(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)


def safe_override(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.mkdir(directory)


def check_transport_activation(lib):
    # Operate on the newlib, should arrive in the 99c-31c format
    errmsg = """
 Please define the pair activation-transport lib for the SDDR benchmark
 (e.g. 99c-31c). See additional details in the documentation.
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
