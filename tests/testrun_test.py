"""

@author: Davide Laghi

Copyright 2022, the JADE Development Team. All rights reserved.

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
import sys
import os
import pandas as pd
from shutil import rmtree

from jade.testrun import Test, SphereTest, SphereTestSDDR, MultipleTest
from f4enix.input.libmanager import LibManager
from f4enix.input.MCNPinput import D1S_Input
import pytest
import json
from jade.__version__ import __version__
from jade.__openmc__ import OMC_AVAIL
from f4enix.input.d1suned import ReactionFile

cp = os.path.dirname(os.path.abspath(__file__))
# TODO change this using the files and resources support in Python>10
root = os.path.dirname(cp)
sys.path.insert(1, root)

# Get a libmanager
ACTIVATION_FILE = os.path.join(cp, "TestFiles", "libmanager", "Activation libs.xlsx")
XSDIR_FILE = os.path.join(cp, "TestFiles", "libmanager", "xsdir")
ISOTOPES_FILE = os.path.join(root, "jade", "resources", "Isotopes.txt")
XSDIR31c_OPENMC = os.path.join(cp, "TestFiles", "libmanager", "31c.xml")
XSDIR31c_SERPENT = os.path.join(cp, "TestFiles", "libmanager", "xsdir.serp")

# Useful files
FILES = os.path.join(cp, "TestFiles", "testrun")


@pytest.fixture
def LM():
    df_rows = [
        ["99c", "sda", "", XSDIR_FILE, XSDIR_FILE, None, None],
        ["98c", "acsdc", "", XSDIR_FILE, XSDIR_FILE, None, None],
        ["21c", "adsadsa", "", XSDIR_FILE, None, None, None],
        ["31c", "adsadas", "", XSDIR_FILE, None, XSDIR31c_OPENMC, XSDIR31c_SERPENT],
        ["00c", "sdas", "", XSDIR_FILE, None, None, None],
        ["71c", "sdasxcx", "", XSDIR_FILE, None, None, None],
        ["81c", "sdasxcx", "yes", XSDIR_FILE, None, None, None],
        ["93c", "sda", "", XSDIR_FILE, XSDIR_FILE, None, None],
    ]
    df_lib = pd.DataFrame(df_rows)
    df_lib.columns = ["Suffix", "Name", "Default", "MCNP", "d1S", "OpenMC", "Serpent"]

    return LibManager(
        df_lib, activationfile=ACTIVATION_FILE, isotopes_file=ISOTOPES_FILE
    )


class TestTest:
    files = os.path.join(FILES, "Test")
    dummyout = os.path.join(FILES, "dummy")

    def test_build_normal(self, LM: LibManager, tmpdir):
        # Just check that nothing breaks
        lib = "81c"
        inp_name = "ITER_1D"
        inp = os.path.join(self.files, inp_name)
        config_data = {
            "Description": "dummy",
            "Folder Name": inp_name,
            "OnlyInput": True,
            "Post-Processing": False,
            "NPS cut-off": 10,
            # "CTME cut-off": 10,
            # "Relative Error cut-off": "F1-0.1",
            "Custom Input": 2,
            "MCNP": True,
            "OpenMC": OMC_AVAIL,
            "Serpent": True,
        }
        config = pd.Series(config_data)
        VRTpath = "dummy"
        conf_path = "dummy"

        # Build the test
        test = Test(inp, lib, config, conf_path, "c", "dummy")
        test.generate_test(tmpdir, LM)
        metadata_file = os.path.join(tmpdir, "ITER_1D", "mcnp", "metadata.json")
        assert os.path.exists(metadata_file)
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        assert metadata["jade_run_version"] == __version__
        assert metadata["benchmark_version"] == "1.0"

    def test_build_d1s(self, LM: LibManager, tmpdir):
        # Just check that nothing breaks
        lib = "99c-31c"
        inp_name = "ITER_Cyl_SDDR"
        inp = os.path.join(self.files, inp_name)
        config_data = {
            "Description": "dummy",
            "Folder Name": inp_name,
            "OnlyInput": True,
            "Run": False,
            "Post-Processing": False,
            "NPS cut-off": 10,
            "CTME cut-off": None,
            "Relative Error cut-off": None,
            "Custom Input": 2,
            "d1S": True,
        }
        config = pd.Series(config_data)
        # VRTpath = os.path.join(self.files, "VRT")
        conf_path = os.path.join(self.files, "ITER_Cyl_SDDR_cnf")

        # Build the test
        test = Test(inp, lib, config, conf_path, "c", "dummy")
        test.generate_test(tmpdir, LM)
        translated_inp = D1S_Input.from_input(
            os.path.join(tmpdir, "ITER_Cyl_SDDR", "d1s", "ITER_Cyl_SDDR"),
            os.path.join(tmpdir, "ITER_Cyl_SDDR", "d1s", "irrad"),
            os.path.join(tmpdir, "ITER_Cyl_SDDR", "d1s", "react"),
        )
        assert translated_inp.reac_file.reactions[2].parent == "25055.99c"
        assert (
            translated_inp.materials["M1"].submaterials[0].zaidList[31].name
            == translated_inp.reac_file.reactions[2].parent
        )
        assert True


class TestSphereTest:
    files = os.path.join(FILES, "SphereTest")
    dummyout = os.path.join(FILES, "dummy")

    def test_build(self, LM: LibManager, tmpdir):
        # Just check that nothing breaks
        lib = "31c"
        inp_name = "Sphere"
        inp = os.path.join(self.files, inp_name)
        config_data = {
            "Description": "dummy",
            "File Name": inp_name,
            "OnlyInput": True,
            "Post-Processing": False,
            "NPS cut-off": 10,
            "CTME cut-off": None,
            "Relative Error cut-off": None,
            "Custom Input": 3,
            "MCNP": True,
            "OpenMC": OMC_AVAIL,
            "Serpent": True,
        }
        config = pd.Series(config_data)
        conf_path = os.path.join(self.files, "Spherecnf")

        # Build the test
        test = SphereTest(inp, lib, config, conf_path, "c", "dummy")
        test.generate_test(tmpdir, LM)
        metadata_file = os.path.join(
            tmpdir, "Sphere", "Sphere_1001_H-1", "mcnp", "metadata.json"
        )
        assert os.path.exists(metadata_file)
        if OMC_AVAIL:
            metadata_file = os.path.join(
                tmpdir, "Sphere", "Sphere_1001_H-1", "openmc", "metadata.json"
            )
            assert os.path.exists(metadata_file)

        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        assert metadata["jade_run_version"] == __version__
        assert metadata["benchmark_version"] == "1.0"


class TestSphereTestSDDR:
    files = os.path.join(FILES, "SphereTestSDDR")

    def test_build(self, LM: LibManager, tmpdir):
        # Just check that nothing breaks
        lib = "93c-31c"
        inp_name = "SphereSDDR"
        inp = os.path.join(self.files, inp_name)
        config_data = {
            "Description": "dummy",
            "File Name": inp_name,
            "OnlyInput": True,
            "Post-Processing": False,
            "NPS cut-off": 10,
            # "CTME cut-off": None,
            # "Relative Error cut-off": None,
            "Custom Input": 3,
            "d1S": True,
        }
        config = pd.Series(config_data)
        # VRTpath = "dummy"
        conf_path = os.path.join(self.files, "cnf")

        # Build the test
        test = SphereTestSDDR(inp, lib, config, conf_path, "c", "dummy")
        test.generate_test(tmpdir, LM)
        # Ensure only one channel is used
        reac_file = os.path.join(
            tmpdir, "SphereSDDR", "SphereSDDR_12025_Mg-25_28", "d1s", "react"
        )
        reac_file = ReactionFile.from_text(reac_file)
        print(reac_file.reactions)
        assert len(reac_file.reactions) == 1

        # check the correct production of parent tracking
        file = os.path.join(
            tmpdir, "SphereSDDR", "SphereSDDR_M101", "d1s", "SphereSDDR_M101_"
        )
        inp = D1S_Input.from_input(file)
        assert len(inp.other_data["FU104"].lines) > 5

        # Check metadata production
        metadata_file = os.path.join(
            tmpdir, "SphereSDDR", "SphereSDDR_M101", "d1s", "metadata.json"
        )
        assert os.path.exists(metadata_file)

        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        assert metadata["jade_run_version"] == __version__
        assert metadata["benchmark_version"] == "1.0"

        assert True


class TestMultipleTest:
    files = os.path.join(FILES, "MultipleTest")
    dummyout = os.path.join(FILES, "dummy")

    def test_build(self, LM: LibManager, tmpdir):
        # Just check that nothing breaks
        lib = "31c"
        # inp_folder = os.path.join(self.files, "Inputs")
        inp_name = "Oktavian"
        inp = os.path.join(self.files, inp_name)
        config_data = {
            "Description": "dummy",
            "Folder Name": inp_name,
            "OnlyInput": True,
            "Post-Processing": False,
            "NPS cut-off": 10,
            # "CTME cut-off": None,
            # "Relative Error cut-off": None,
            "Custom Input": 3,
            "MCNP": True,
            "OpenMC": OMC_AVAIL,
            "Serpent": True,
        }
        config = pd.Series(config_data)
        # VRTpath = "dummy"
        conf_path = os.path.join(self.files, "cnf")

        # Build the test
        test = MultipleTest(inp, lib, config, conf_path, "c", "dummy")
        test.generate_test(tmpdir, LM)
        metadata_file = os.path.join(
            tmpdir, "Oktavian", "Oktavian_Al", "mcnp", "metadata.json"
        )
        assert os.path.exists(metadata_file)

        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        assert metadata["jade_run_version"] == __version__
        assert metadata["benchmark_version"] == "1.0"
