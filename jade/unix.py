# -*- coding: utf-8 -*-

# Created on Mon Nov  4 16:52:09 2019

# @author: JADE Team

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
import re
import subprocess
import sys

if "MODULESHOME" in os.environ:
    if "MODULEPATH" not in os.environ and "win" not in sys.platform:
        f = open(os.environ["MODULESHOME"] + "/init/.modulespath", "r")
        path = []
        for line in f.readlines():
            line = re.sub("#.*$", "", line)
            if line != "":
                path.append(line)
        os.environ["MODULEPATH"] = ":".join(path)

    if "LOADEDMODULES" not in os.environ and "win" not in sys.platform:
        os.environ["LOADEDMODULES"] = ""

def module(*args: list) -> None:
    """Runs the module command using subprocess"""
    if type(args[0]) == type([]):
        args = args[0]
    else:
        args = list(args)
    (output, error) = subprocess.Popen(
        ["/usr/bin/modulecmd", "python"] + args, stdout=subprocess.PIPE
    ).communicate()
    exec(output)


def export(*args: list) -> None:
    """Export environment variables based on input values."""
    if type(args[0]) == type([]):
        args = args[0]
    else:
        args = list(args)
    args = "".join(args)
    environ, value = args.split("=")
    if environ not in os.environ:
        os.environ[environ] = ""
    value = value.replace("$" + environ, os.environ[environ])
    if "$" in value:
        parts = value.split("$")
        for string in parts[1:]:
            parent = string[: string.find("/")]
            value = value.replace("$" + parent, os.environ[parent])
    os.environ[environ] = value


def configure(config_file: str) -> None:
    """Perform configuration based on contents of configuration file.

    Parameters
    ----------
    config_file : str
        The configuration file path.
    """
    with open(config_file, "r") as f:
        for line in f:
            if line.startswith("module"):
                args = line.split()
                module(args[1:])
            if line.startswith("export"):
                args = line.split()
                export(args[1:])
