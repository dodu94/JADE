# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 12:53:52 2021

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

import json
import os
import sys

import pytest

cp = os.path.dirname(os.path.abspath(__file__))
modules_path = os.path.dirname(cp)
sys.path.insert(1, modules_path)

import pandas as pd
from f4enix.input.libmanager import LibManager

import jade.output as output

# from jade.expoutput import SpectrumOutput
import jade.sphereoutput as sout
from jade.__openmc__ import OMC_AVAIL
from jade.__version__ import __version__
from jade.configuration import Configuration
from jade.output import MCNPSimOutput
from jade.postprocess import compareBenchmark

if OMC_AVAIL:
    import jade.openmc as omc

# Files
OUTP_SDDR = os.path.join(
    cp, "TestFiles", "sphereoutput", "SphereSDDR_11023_Na-23_102_o"
)
OUTM_SDDR = os.path.join(
    cp, "TestFiles", "sphereoutput", "SphereSDDR_11023_Na-23_102_m"
)


class TestSphereSDDRMCNPSimOutput:
    out = sout.SphereSDDRMCNPOutput(OUTM_SDDR, OUTP_SDDR)

    def test_get_single_excel_data(self):
        vals, errors = self.out.get_single_excel_data()
        assert isinstance(vals, pd.Series)
        assert isinstance(errors, pd.Series)
        assert len(vals) == 23
        assert len(errors) == 23


class TestMCNPSimOutput:
    def test_mcnpoutput(self):
        out = MCNPSimOutput(OUTM_SDDR, OUTP_SDDR)
        t4 = out.tallydata[4]
        t2 = out.tallydata[2]
        assert list(t4.columns) == ["Cells", "Segments", "Value", "Error"]
        assert len(t4) == 1
        assert len(t2) == 176
        assert list(t2.columns) == ["Energy", "Value", "Error"]


class MockLog:
    def __init__(self) -> None:
        self.log = []

    def adjourn(self, message: str) -> None:
        self.log.append(message)


class MockSession:
    def __init__(self, conf: Configuration, root_dir: os.PathLike) -> None:
        self.state = "dummy"
        self.path_templates = os.path.join(modules_path, "jade", "templates")
        self.path_cnf = os.path.join(
            modules_path,
            "jade",
            "default_settings",
            "Benchmarks_Configuration",
        )
        self.path_run = os.path.join(cp, "TestFiles", "output", "Simulations")
        self.conf = conf
        self.path_comparison = root_dir.mkdir("comparison")
        self.path_single = root_dir.mkdir("single")
        self.path_exp_res = os.path.join(cp, "TestFiles", "output", "exp_results")
        self.log = MockLog()


class TestBenchmarkOutput:
    def test_single_excel_mcnp(self, tmpdir):
        conf = Configuration(
            os.path.join(cp, "TestFiles", "output", "config_test.xlsx")
        )
        session = MockSession(conf, tmpdir)
        out = output.MCNPBenchmarkOutput("32c", "mcnp", "ITER_1D", session)
        out._generate_single_excel_output()
        out._print_raw()

        assert os.path.exists(
            os.path.join(
                session.path_single,
                "32c",
                "ITER_1D",
                "mcnp",
                "Excel",
                "ITER_1D_32c.xlsx",
            )
        )
        metadata_path = os.path.join(
            session.path_single, "32c", "ITER_1D", "mcnp", "Raw_Data", "metadata.json"
        )
        assert os.path.exists(metadata_path)
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        assert metadata["jade_run_version"] == "0.0.1"
        assert metadata["jade_version"] == __version__
        assert metadata["code_version"] == "6.2"

    @pytest.mark.skipif(not OMC_AVAIL, reason="OpenMC is not available")
    def test_single_excel_openmc(self, tmpdir):
        conf = Configuration(
            os.path.join(cp, "TestFiles", "output", "config_test.xlsx")
        )
        session = MockSession(conf, tmpdir)
        out = output.OpenMCBenchmarkOutput("32c", "openmc", "ITER_1D", session)
        out._generate_single_excel_output()
        out._print_raw()

        assert os.path.exists(
            os.path.join(
                session.path_single,
                "32c",
                "ITER_1D",
                "openmc",
                "Excel",
                "ITER_1D_32c.xlsx",
            )
        )
        metadata_path = os.path.join(
            session.path_single, "32c", "ITER_1D", "openmc", "Raw_Data", "metadata.json"
        )
        assert os.path.exists(metadata_path)
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        assert metadata["jade_run_version"] == "0.0.1"
        assert metadata["jade_version"] == __version__
        assert metadata["code_version"] == "0.14.0"

    def test_iter_cyl(self, tmpdir):
        conf = Configuration(
            os.path.join(cp, "TestFiles", "output", "config_itercyl.xlsx")
        )
        session = MockSession(conf, tmpdir)
        out = output.MCNPBenchmarkOutput("99c", "d1s", "ITER_Cyl_SDDR", session)
        out.single_postprocess()
        out = output.MCNPBenchmarkOutput("93c", "d1s", "ITER_Cyl_SDDR", session)
        out.single_postprocess()
        compareBenchmark(session, "99c-93c", "d1s", ["ITER_Cyl_SDDR"], exp=False)


# class TestExperimentalOutput:

#     def test_print_raw_metadata(self, tmpdir):
#         conf = Configuration(
#             os.path.join(cp, "TestFiles", "output", "config_test.xlsx")
#         )
#         session = MockSession(conf, tmpdir)
#         out = SpectrumOutput(
#             ["Exp", "32c"], "mcnp", "Oktavian", session, multiplerun=True
#         )
#         out._extract_outputs()
#         out._read_exp_results()
#         out._print_raw()

#         for folder in os.listdir(session.path_comparison):
#             path = os.path.join(
#                 session.path_comparison,
#                 folder,
#                 "Oktavian",
#                 "mcnp",
#                 "Raw_Data",
#                 "32c",
#                 "metadata.json",
#             )

#             assert os.path.exists(path)
#             with open(path, "r") as f:
#                 metadata = json.load(f)
#             assert metadata["jade_run_version"] == "0.0.1"
#             assert metadata["jade_version"] == __version__
