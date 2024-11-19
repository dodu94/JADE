"""

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

import os
import sys

import pytest

from jade.configuration import BenchmarkConfiguration, Configuration

cp = os.path.dirname(os.path.abspath(__file__))
modules_path = os.path.dirname(cp)
sys.path.insert(1, modules_path)

resources = os.path.join(cp, "TestFiles", "configuration")
MAIN_CONFIG_FILE = os.path.join(resources, "mainconfig.xlsx")


@pytest.fixture
def config():
    configob = Configuration(MAIN_CONFIG_FILE)
    return configob


class TestConfiguration:
    def test_read(self, config):
        # TODO
        # Check that everything is read in a correct way
        assert config.mpi_tasks == 4

    def test_get_lib_name(self, config):
        suffix_list = ["21c", "33c", "pincopalle"]
        expected_list = ["FENDL 2.1c", "33c", "pincopalle"]
        for suffix, expected in zip(suffix_list, expected_list):
            assert config.get_lib_name(suffix) == expected


class TestComputationalConfig:
    def test_conformity_defaults(self):
        """test all yaml configuration files found in the default settings"""
        root = os.path.join(
            modules_path, "jade", "default_settings", "Benchmarks_Configuration"
        )
        for file in os.listdir(root):
            if file.endswith(".yaml"):
                cfg = BenchmarkConfiguration.from_yaml(os.path.join(root, file))
                assert cfg

    def test_allowables(self):
        cfg = BenchmarkConfiguration.from_yaml(os.path.join(resources, "ITER_1D.yaml"))
        # ensure that ints are correctly converted
        assert cfg.excel_options[44]
        # additional keyword not supported by the data class
        with pytest.raises(TypeError):
            cfg = BenchmarkConfiguration.from_yaml(
                os.path.join(resources, "wrong_cfg.yaml")
            )

        # unsupported plot type
        with pytest.raises(ValueError):
            cfg = BenchmarkConfiguration.from_yaml(
                os.path.join(resources, "wrong_cfg2.yaml")
            )

        # unsupported Tally bin type
        with pytest.raises(ValueError):
            cfg = BenchmarkConfiguration.from_yaml(
                os.path.join(resources, "wrong_cfg3.yaml")
            )
