import os
import sys

import pandas as pd
import pytest

from jade.configuration import Configuration
from jade.main import Session

cp = os.path.dirname(os.path.abspath(__file__))
modules_path = os.path.dirname(cp)
sys.path.insert(1, modules_path)

resources = os.path.join(cp, "TestFiles", "expoutput")
from f4enix.input.libmanager import LibManager

import jade.expoutput as expoutput
import jade.output as outp

root = os.path.dirname(cp)
CONFIG_FILE_EXP = os.path.join(resources, "mainconfig.xlsx")
ACTIVATION_FILE = os.path.join(cp, "TestFiles", "libmanager", "Activation libs.xlsx")
XSDIR_FILE = os.path.join(cp, "TestFiles", "libmanager", "xsdir")
ISOTOPES_FILE = os.path.join(root, "jade", "resources", "Isotopes.txt")


# I don't want to deal with testing the Session object itself for the moment
class MockUpSession(Session):
    def __init__(self, tmpdir, lm: LibManager):
        self.conf = Configuration(CONFIG_FILE_EXP)
        self.path_comparison = os.path.join(tmpdir, "Post-Processing", "Comparison")
        self.path_single = os.path.join(tmpdir, "Post-Processing", "Single_Libraries")
        self.path_exp_res = os.path.join(resources, "ExperimentalResults")
        self.path_pp = os.path.join(tmpdir, "Post-Processing")
        self.path_run = os.path.join(resources, "Simulations")
        self.path_test = resources
        self.state = None
        self.path_templates = os.path.join(resources, "templates")
        self.path_cnf = os.path.join(resources, "Benchmarks_Configuration")
        self.path_quality = None
        self.path_uti = None
        self.path_comparison = os.path.join(tmpdir, "Post-Processing", "Comparisons")
        self.lib_manager = lm


# class TestExpOutput:
#     @pytest.fixture()
#     def session_mock(self, tmpdir, lm: LibManager):
#         session = MockUpSession(tmpdir, lm)
#         return session

#     @pytest.fixture
#     def lm(self):
#         df_rows = [
#             ["99c", "sda", "", XSDIR_FILE],
#             ["98c", "acsdc", "", XSDIR_FILE],
#             ["21c", "adsadsa", "", XSDIR_FILE],
#             ["31c", "adsadas", "", XSDIR_FILE],
#             ["00c", "sdas", "yes", XSDIR_FILE],
#             ["71c", "sdasxcx", "", XSDIR_FILE],
#         ]
#         df_lib = pd.DataFrame(df_rows)
#         df_lib.columns = ["Suffix", "Name", "Default", "MCNP"]

#         return LibManager(
#             df_lib, activationfile=ACTIVATION_FILE, isotopes_file=ISOTOPES_FILE
#         )

#     def test_benchmarkoutput(self, session_mock: MockUpSession):
#         code = "mcnp"
#         testname = "ITER_1D"
#         os.makedirs(session_mock.path_comparison)
#         os.makedirs(session_mock.path_single)
#         self.benchoutput_32c = outp.MCNPBenchmarkOutput(
#             "32c", code, testname, session_mock
#         )
#         self.benchoutput_32c.single_postprocess()
#         self.benchoutput_31c = outp.MCNPBenchmarkOutput(
#             "31c", code, testname, session_mock
#         )
#         self.benchoutput_31c.single_postprocess()
#         self.benchoutput_comp = outp.MCNPBenchmarkOutput(
#             ["32c", "31c"], code, testname, session_mock
#         )
#         self.benchoutput_comp.compare()
#         assert self.benchoutput_31c.raw_data[24]["Error"].iloc[10] == 0.7602
#         assert self.benchoutput_31c.raw_data[94]["Cells"].iloc[2] == 54
#         assert True

#     def test_benchmarkoutputmesh(self, session_mock: MockUpSession):
#         code = "mcnp"
#         testname = "WCLL_TBM_1D"
#         os.makedirs(session_mock.path_comparison)
#         os.makedirs(session_mock.path_single)
#         self.benchoutput_32c = outp.MCNPBenchmarkOutput(
#             "32c", code, testname, session_mock
#         )
#         self.benchoutput_32c.single_postprocess()
#         self.benchoutput_31c = outp.MCNPBenchmarkOutput(
#             "31c", code, testname, session_mock
#         )
#         self.benchoutput_31c.single_postprocess()
#         self.benchoutput_comp = outp.MCNPBenchmarkOutput(
#             ["32c", "31c"], code, testname, session_mock
#         )
#         self.benchoutput_comp.compare()
#         assert self.benchoutput_31c.raw_data[214]["Error"].iloc[74] == 0.028364
#         assert self.benchoutput_31c.raw_data[244]["Cor A"].iloc[78] == 1084.20
#         assert True

#     def test_spectrumoutput(self, session_mock: MockUpSession):
#         code = "mcnp"
#         os.makedirs(session_mock.path_comparison)
#         os.makedirs(session_mock.path_single)
#         testname = "Oktavian"
#         self.benchoutput_comp = expoutput.SpectrumOutput(
#             ["32c", "31c"], code, testname, session_mock, multiplerun=True
#         )
#         self.benchoutput_comp.compare()
#         testname = "FNS-TOF"
#         self.benchoutput_comp = expoutput.MultipleSpectrumOutput(
#             ["32c", "31c", "00c"], code, testname, session_mock, multiplerun=True
#         )
#         self.benchoutput_comp.compare()
#         # check that all the raw data is dumped
#         path2raw = os.path.join(
#             session_mock.path_comparison,
#             "32c_Vs_31c_Vs_00c",
#             testname,
#             "mcnp",
#             "Raw_Data",
#         )
#         assert (
#             self.benchoutput_comp.raw_data[("Be-15", "32c")][5]["Energy"].iloc[10]
#             == 0.0196645
#         )
#         assert (
#             self.benchoutput_comp.raw_data[("Be-15", "32c")][5]["Value"].iloc[14]
#             == 3.93654e-08
#         )
#         assert len(os.listdir(path2raw)) == 3

#         testname = "ASPIS-PCA-Replica_flux"
#         self.benchoutput_comp = expoutput.SpectrumOutput(
#             ["32c", "31c", "00c"], code, testname, session_mock, multiplerun=True
#         )
#         self.benchoutput_comp.compare()
#         # check that all the raw data is dumped
#         path2raw = os.path.join(
#             session_mock.path_comparison,
#             "32c_Vs_31c_Vs_00c",
#             testname,
#             "mcnp",
#             "Raw_Data",
#         )
#         assert (
#             self.benchoutput_comp.raw_data[("spectra", "32c")][914]["Energy"].iloc[10]
#             == 1.83000e-01
#         )
#         assert (
#             self.benchoutput_comp.raw_data[("spectra", "00c")][924]["Error"].iloc[0]
#             == 0.3742
#         )
#         assert len(os.listdir(path2raw)) == 3

#     def test_shieldingoutput(self, session_mock: MockUpSession):
#         code = "mcnp"
#         testname = "FNG-W"
#         os.makedirs(session_mock.path_comparison)
#         os.makedirs(session_mock.path_single)
#         self.benchoutput_comp = expoutput.ShieldingOutput(
#             ["32c", "31c", "00c"], code, testname, session_mock, multiplerun=True
#         )
#         self.benchoutput_comp.compare()
#         # check that all the raw data is dumped
#         # check that all the raw data is dumped
#         path2raw = os.path.join(
#             session_mock.path_comparison,
#             "32c_Vs_31c_Vs_00c",
#             testname,
#             "mcnp",
#             "Raw_Data",
#         )
#         assert self.benchoutput_comp.raw_data[("Au", "31c")][4]["Cells"].iloc[1] == 643
#         assert (
#             self.benchoutput_comp.raw_data[("Au", "31c")][4]["Error"].iloc[2] == 0.6207
#         )
#         assert len(os.listdir(path2raw)) == 3

#         testname = "FNG-SS"
#         self.benchoutput_comp = expoutput.ShieldingOutput(
#             ["32c", "31c", "00c"], code, testname, session_mock, multiplerun=True
#         )
#         self.benchoutput_comp.compare()
#         # check that all the raw data is dumped
#         # check that all the raw data is dumped
#         path2raw = os.path.join(
#             session_mock.path_comparison,
#             "32c_Vs_31c_Vs_00c",
#             testname,
#             "mcnp",
#             "Raw_Data",
#         )
#         assert self.benchoutput_comp.raw_data[("Al", "31c")][4]["Cells"].iloc[1] == 501
#         assert (
#             self.benchoutput_comp.raw_data[("Au", "00c")][4]["Error"].iloc[2] == 0.0194
#         )

#         testname = "FNG-SiC"
#         self.benchoutput_comp = expoutput.ShieldingOutput(
#             ["32c", "31c", "00c"], code, testname, session_mock, multiplerun=True
#         )
#         self.benchoutput_comp.compare()
#         # check that all the raw data is dumped
#         # check that all the raw data is dumped
#         path2raw = os.path.join(
#             session_mock.path_comparison,
#             "32c_Vs_31c_Vs_00c",
#             testname,
#             "mcnp",
#             "Raw_Data",
#         )
#         assert self.benchoutput_comp.raw_data[("Al", "31c")][4]["Cells"].iloc[1] == 612
#         assert (
#             self.benchoutput_comp.raw_data[("TLD", "00c")][16]["Error"].iloc[2]
#             == 0.0089
#         )

#         testname = "ASPIS-PCA-Replica_RR"
#         self.benchoutput_comp = expoutput.ShieldingOutput(
#             ["32c", "31c", "00c"], code, testname, session_mock, multiplerun=True
#         )
#         self.benchoutput_comp.compare()
#         # check that all the raw data is dumped
#         # check that all the raw data is dumped
#         path2raw = os.path.join(
#             session_mock.path_comparison,
#             "32c_Vs_31c_Vs_00c",
#             testname,
#             "mcnp",
#             "Raw_Data",
#         )
#         assert self.benchoutput_comp.raw_data[("In", "31c")][4]["Cells"].iloc[1] == 924
#         assert (
#             self.benchoutput_comp.raw_data[("Rh", "00c")][4]["Error"].iloc[2] == 0.0049
#         )

#     def test_tiaraoutput(self, session_mock: MockUpSession):
#         code = "mcnp"
#         testname = "Tiara-BS"
#         os.makedirs(session_mock.path_comparison)
#         os.makedirs(session_mock.path_single)
#         self.benchoutput_comp = expoutput.TiaraBSOutput(
#             ["32c", "31c"], code, testname, session_mock, multiplerun=True
#         )
#         self.benchoutput_comp.compare()
#         # conf = config.iloc[4]
#         assert (
#             self.benchoutput_comp.raw_data[("cc-43-25-40", "32c")][14]["Value"].iloc[0]
#             == 448.291
#         )
#         assert (
#             self.benchoutput_comp.raw_data[("cc-43-25-40", "32c")][24]["Error"].iloc[0]
#             == 0.5933
#         )
#         testname = "Tiara-FC"
#         self.benchoutput_comp = expoutput.TiaraFCOutput(
#             ["32c", "31c"], code, testname, session_mock, multiplerun=True
#         )
#         self.benchoutput_comp.compare()
#         # check that all the raw data is dumped
#         path2raw = os.path.join(
#             session_mock.path_comparison,
#             "32c_Vs_31c",
#             testname,
#             "mcnp",
#             "Raw_Data",
#         )
#         assert (
#             self.benchoutput_comp.raw_data[("cc-43-25-00", "32c")][24]["Value"].iloc[0]
#             == 6.02648e03
#         )
#         assert (
#             self.benchoutput_comp.raw_data[("cc-43-25-00", "32c")][44]["Value"].iloc[0]
#             == 4.59843e02
#         )
#         assert len(os.listdir(path2raw)) == 2
#         assert True

#     def test_fngoutput(self, session_mock: MockUpSession):
#         code = "d1s"
#         testname = "FNG"
#         os.makedirs(session_mock.path_comparison)
#         os.makedirs(session_mock.path_single)
#         self.benchoutput_comp = expoutput.FNGOutput(
#             ["99c", "98c"], code, testname, session_mock, multiplerun=True
#         )
#         self.benchoutput_comp.compare()
#         # check that all the raw data is dumped
#         path2raw = os.path.join(
#             session_mock.path_comparison,
#             "99c_Vs_98c",
#             testname,
#             "d1s",
#             "Raw_Data",
#         )
#         assert len(os.listdir(path2raw)) == 2
#         assert True

#     def test_fnghcpboutput(self, session_mock: MockUpSession):
#         code = "mcnp"
#         testname = "FNG-HCPB"
#         os.makedirs(session_mock.path_comparison)
#         os.makedirs(session_mock.path_single)
#         self.benchoutput_comp = expoutput.FNGHCPBOutput(
#             ["32c", "31c", "00c"], code, testname, session_mock, multiplerun=True
#         )
#         self.benchoutput_comp.compare()
#         # check that all the raw data is dumped
#         # check that all the raw data is dumped
#         path2raw = os.path.join(
#             session_mock.path_comparison,
#             "32c_Vs_31c_Vs_00c",
#             testname,
#             "mcnp",
#             "Raw_Data",
#         )
#         assert (
#             self.benchoutput_comp.raw_data[("Al", "31c")][4]["Cells"].iloc[1] == 48002
#         )
#         assert (
#             self.benchoutput_comp.raw_data[("H3", "00c")][84]["Error"].iloc[2] == 0.0165
#         )
#         assert len(os.listdir(path2raw)) == 3


class TestMCNPExpBenchmarkOutput:
    @pytest.fixture()
    def session_mock(self, tmpdir, lm: LibManager):
        session = MockUpSession(tmpdir, lm)
        return session

    @pytest.fixture
    def lm(self):
        df_rows = [
            ["99c", "sda", "", XSDIR_FILE],
            ["98c", "acsdc", "", XSDIR_FILE],
            ["21c", "adsadsa", "", XSDIR_FILE],
            ["31c", "adsadas", "", XSDIR_FILE],
            ["00c", "sdas", "yes", XSDIR_FILE],
            ["71c", "sdasxcx", "", XSDIR_FILE],
        ]
        df_lib = pd.DataFrame(df_rows)
        df_lib.columns = ["Suffix", "Name", "Default", "MCNP"]

        return LibManager(
            df_lib, activationfile=ACTIVATION_FILE, isotopes_file=ISOTOPES_FILE
        )

    def test_init(self, session_mock: MockUpSession):
        # TODO add a test here where you just initiliaze the object without errors
        benchoutput = expoutput.MCNPExpBenchmarkOutput(
            "32c", "mcnp", "Okatavian", session_mock, multiplerun=True
        )

    def test_pp_excel_comparison(self):
        # TODO verify that the CE tables are correctly dumped
        # benchoutput = expoutput.MCNPExpBenchmarkOutput(
        #     "32c", "mcnp", "Okatavian", session_mock, multiplerun=True
        # )
        # benchoutput.single_postprocess()
        pass
