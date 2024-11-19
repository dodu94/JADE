# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 16:52:36 2019

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
import datetime
import os
import re
import sys

import jade.testrun as testrun
import logging


def executeBenchmarksRoutines(session, lib: str, runoption: str, exp=False) -> None:
    """
    Check which benchmarks have to be generated and/or run and execute their
    routines

    Parameters
    ----------
    session : jade.Session
        Current JADE session.
    lib : str (or dic string)
        library to assess (e.g. 31c)
        or couple activation+transport (e.g. 99c-31c).
        Double quotes are needed.
    runoption : str
        if 'c' the code is run in console mode, if 's' is submitted as batch.
    exp : bool
        if True the experimental Benchmarks are selected. The default is False

    Returns
    -------
    None.

    """
    # Get the settings for the tests
    if exp:
        config = session.conf.exp_default.set_index("Description")
    else:
        config = session.conf.comp_default.set_index("Description")

    for testname, row in config.iterrows():
        # Check for active test first
        if sys.platform.startswith("win"):
            if bool(row["Serpent"]) or bool(row["OpenMC"]):
                print(
                    f"\n"
                    f"'{testname}' selected. OpenMC and Serpent are not currently supported on Windows."
                    f"\n"
                )
                row["Serpent"] = False
                row["OpenMC"] = False

        if (
            bool(row["OnlyInput"])
            or bool(row["MCNP"])
            or bool(row["Serpent"])
            or bool(row["OpenMC"])
            or bool(row["d1S"])
        ):
            if bool(row["OnlyInput"]) and not any(
                [
                    bool(row["MCNP"]),
                    bool(row["Serpent"]),
                    bool(row["OpenMC"]),
                    bool(row["d1S"]),
                ]
            ):
                if testname in ["SphereSDDR", "FNG", "ITER_Cyl_SDDR"]:
                    row["d1S"] = True
                else:
                    print(
                        "Transport code was not specified or is not available for input generation, defaulting to MCNP"
                    )
                    print("")
                    row["MCNP"] = True

            print("        -- " + testname.upper() + " STARTED --\n")
            print(" Generating input files:" + "    " + str(datetime.datetime.now()))
            logging.info(
                "%s run started     %s", testname.upper(), str(datetime.datetime.now())
            )

            # --- Input Generation ---
            # Collect infos
            libmanager = session.lib_manager

            # Handle dic string as lib
            pat_libs = re.compile(r'"\d\d[a-zA-Z]"')
            if lib[0] == "{":
                libs = pat_libs.findall(lib)
                libpath = libs[1][1:-1]
            elif "-" in lib:
                libpath = lib[:3]
            else:
                libpath = lib

            if testname in [
                "FNG Bulk Blanket and Shielding Experiment",
                "FNG Tungsten",
                "ASPIS Iron-88 benchmark",
                "FNG SiC",
                "FNG HCPB",
                "ASPIS PCA Replica RR",
                "FNG Stainless Steel",
            ]:
                var = {"00c": lib, "34y": "34y"}
            else:
                var = lib

            # get path to libdir
            outpath = os.path.join(session.path_run, libpath)
            safemkdir(outpath)
            fname = row["Folder Name"]
            inppath = os.path.join(session.path_inputs, fname)
            #            VRTpath = os.path.join(session.path_inputs, "ITER_Cyl_SDDR", "d1S")
            confpath = os.path.join(session.path_cnf, fname.split(".")[0])

            # Generate test
            lib_name = session.conf.get_lib_name(var)
            args = (inppath, var, row, confpath, runoption, lib_name)
            # Handle special cases
            if testname == "Sphere Leakage Test":
                test = testrun.SphereTest(*args)

            elif testname == "Sphere SDDR":
                test = testrun.SphereTestSDDR(*args)

            elif fname in [
                "Oktavian",
                "Tiara-BC",
                "Tiara-BS",
                "Tiara-FC",
                "FNS-TOF",
                "FNG-BKT",
                "FNG-W",
                "ASPIS-Fe88",
                "TUD-Fe",
                "TUD-W",
                "FNG-SiC",
                "FNG-HCPB",
                "ASPIS-PCA-Replica_flux",
                "ASPIS-PCA-Replica_RR",
                "FNG-SS",
            ]:
                test = testrun.MultipleTest(*args)

            elif fname == "FNG":
                test = testrun.MultipleTest(*args, TestOb=testrun.FNGTest)

            else:
                test = testrun.Test(*args)

            # write the input(s)
            if testname in ["Sphere Leakage Test", "Sphere SDDR"]:
                try:
                    limit = int(row["Custom Input"])
                except ValueError:
                    # go back do the default which is None
                    limit = None
                test.generate_test(outpath, libmanager, limit=limit)
            else:
                test.generate_test(outpath, libmanager)
            # Adjourn log
            logging.info(
                "%s test input generated with success      %s",
                testname.upper(),
                str(datetime.datetime.now()),
            )

            if bool(row["OnlyInput"]):
                print("\n        -- " + testname.upper() + " COMPLETED --\n")
                logging.info("Only inputs were generated for %s", testname.upper())
            else:
                # --- Input Run ---
                text = " Simulation running:         " + str(datetime.datetime.now())
                print(text)
                logging.info(text)
                # test.run(cpu=session.conf.cpu)
                test.run(session.conf, session.lib_manager, runoption)
                print("\n        -- " + testname.upper() + " COMPLETED --\n")
                # Adjourn log
                logging.info(
                    "%s run completed with success    %s",
                    testname.upper(),
                    str(datetime.datetime.now()),
                )


def safemkdir(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)
