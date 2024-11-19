# -*- coding: utf-8 -*-
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
import shutil
import sys

import numpy as np
import pytest

cp = os.path.dirname(os.path.abspath(__file__))
modules_path = os.path.dirname(cp)
sys.path.insert(1, modules_path)

import jade.plotter as plotter
from jade.plotter import Plotter, PlotType

OUTPATH = os.path.join(cp, "tmp")
outname = "dummy"
quantity = "Flux"
unit = "#/cc"
xlabel = "xlabel"
title = "plot title"
x = np.random.rand(50)
data1 = {
    "x": x,
    "y": np.random.rand(50) * 100,
    "err": np.random.rand(50),
    "ylabel": "data1",
}
data2 = {
    "x": x,
    "y": np.random.rand(50) * 100,
    "err": np.random.rand(50),
    "ylabel": "data2",
}
data = [data1, data2]
default_testname = "dummy"
xlimits = (np.sort(x)[3], np.sort(x)[-5])
vlines = {"major": [np.sort(x)[10]], "minor": [np.sort(x)[12]]}
add_labels = {"major": [("major label", 0.1)], "minor": [("minor label", 0.5)]}
recs = plotter.TBM_HCPB_RECT

KEYARGS = {
    "data": data,
    "title": title,
    "outpath": OUTPATH,
    "outname": outname,
    "quantity": quantity,
    "unit": unit,
    "xlabel": xlabel,
    "testname": default_testname,
}

AVAILABLE_PLOTS = [
    "Binned graph",
    "Ratio graph",
    "Experimental points",
    "Discrete Experimental points",
    "Grouped bars",
]


class TestPlotter:
    """This will merely checks that the plots can be generated without having
    unexpected error. It does not guarantee that the code works as intended.
    """

    def test_get_limits(self):
        lowerlimits = np.random.rand(10)
        upperlimits = lowerlimits * 3
        print(upperlimits)
        xdata = x
        ydata = data1["y"] / data2["y"]
        for lowerlimit, upperlimit in zip(lowerlimits, upperlimits):
            limits = plotter._get_limits(lowerlimit, upperlimit, ydata, xdata)
            assert len(limits) == 3
            totlen = len(limits[0][0]) + len(limits[1][0]) + len(limits[2][0])
            assert totlen == len(xdata)
            for limit in limits:
                assert len(limit[0]) == len(limit[1])
                assert limit[0].size == limit[1].size

    @pytest.mark.parametrize("plot_type", AVAILABLE_PLOTS)
    def test_plot(self, plot_type):
        plotterob = Plotter(**KEYARGS)
        # In this first one check also the error
        try:
            plotterob.plot("wrongone")
            assert False
        except ValueError:
            assert True

        args = [PlotType(plot_type)]
        keyargs = {}
        self._plot(plotterob, args, keyargs)

    def test_waves(self):
        data1 = {
            "x": x,
            "y": [np.random.rand(50) * 100, np.random.rand(50) * 100],
            "err": np.random.rand(50),
            "ylabel": "data1",
        }
        data2 = {
            "x": x,
            "y": [np.random.rand(50) * 100, np.random.rand(50) * 100],
            "err": np.random.rand(50),
            "ylabel": "data2",
        }
        data = [data1, data2]
        keyargs = KEYARGS.copy()
        keyargs["data"] = data
        keyargs["quantity"] = ["ax a", "ax b"]

        plotterob = Plotter(**keyargs)

        args = [PlotType("Waves")]
        keyargs = {}
        self._plot(plotterob, args, keyargs)

    def test_ratio_special(self):
        plotterob = Plotter(**KEYARGS)
        # Create the tmp directory
        try:
            os.mkdir(OUTPATH)
        except FileExistsError:
            pass

        try:
            keyargs = {
                "additional_labels": add_labels,
                "v_lines": vlines,
                "xlimits": xlimits,
                "recs": recs,
                "markers": True,
            }
            plotterob._ratio_plot(**keyargs)
            assert True
        finally:
            # remove the temporary directory
            shutil.rmtree(OUTPATH)

    def test_grouped_bar_special(self):
        plotterob = Plotter(**KEYARGS)
        # Create the tmp directory
        try:
            os.mkdir(OUTPATH)
        except FileExistsError:
            pass

        try:
            keyargs = {"log": True}
            plotterob._grouped_bar(**keyargs)
            assert True
        finally:
            # remove the temporary directory
            shutil.rmtree(OUTPATH)

    @staticmethod
    def _plot(plotterob, args, keyargs):
        # Create the tmp directory
        try:
            os.mkdir(OUTPATH)
        except FileExistsError:
            pass

        try:
            plotterob.plot(*args, **keyargs)
            assert True
        finally:
            # remove the temporary directory
            shutil.rmtree(OUTPATH)
