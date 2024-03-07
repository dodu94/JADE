import os
import sys
import pytest
import pandas as pd
from jade.main import Session
from jade.configuration import Configuration

cp = os.path.dirname(os.path.abspath(__file__))
modules_path = os.path.dirname(cp)
sys.path.insert(1, modules_path)

resources = os.path.join(cp, "TestFiles", "expoutput")
import jade.expoutput as expoutput
import jade.output as outp
from jade.libmanager import LibManager

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


class TestExpOutput:
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

    def test_benchmarkoutput(self, session_mock: MockUpSession, lm: LibManager):

        config = session_mock.conf.comp_default.set_index("Description")
        conf = config.iloc[1]
        os.makedirs(session_mock.path_comparison)
        os.makedirs(session_mock.path_single)
        self.benchoutput_32c = outp.BenchmarkOutput("32c", conf, session_mock)
        self.benchoutput_32c.single_postprocess()
        self.benchoutput_31c = outp.BenchmarkOutput("31c", conf, session_mock)
        self.benchoutput_31c.single_postprocess()
        self.benchoutput_comp = outp.BenchmarkOutput(["32c", "31c"], conf, session_mock)
        self.benchoutput_comp.compare()
        assert True

    def test_spectrumoutput(self, session_mock: MockUpSession):

        config = session_mock.conf.exp_default.set_index("Description")
        conf = config.iloc[0]
        os.makedirs(session_mock.path_comparison)
        os.makedirs(session_mock.path_single)
        self.benchoutput_comp = expoutput.SpectrumOutput(
            ["32c", "31c"], conf, session_mock, multiplerun=True
        )
        self.benchoutput_comp.compare()
        conf = config.iloc[5]
        self.benchoutput_comp = expoutput.MultipleSpectrumOutput(
            ["32c", "31c"], conf, session_mock, multiplerun=True
        )
        self.benchoutput_comp.compare()
        assert True

    def test_shieldingoutput(self, session_mock: MockUpSession):

        config = session_mock.conf.exp_default.set_index("Description")
        conf = config.iloc[7]
        os.makedirs(session_mock.path_comparison)
        os.makedirs(session_mock.path_single)
        self.benchoutput_comp = expoutput.ShieldingOutput(
            ["32c", "31c"], conf, session_mock, multiplerun=True
        )
        self.benchoutput_comp.compare()
        assert True

    def test_tiaraoutput(self, session_mock: MockUpSession):

        config = session_mock.conf.exp_default.set_index("Description")
        conf = config.iloc[3]
        os.makedirs(session_mock.path_comparison)
        os.makedirs(session_mock.path_single)
        self.benchoutput_comp = expoutput.TiaraBSOutput(
            ["32c", "31c"], conf, session_mock, multiplerun=True
        )
        self.benchoutput_comp.compare()
        conf = config.iloc[4]
        self.benchoutput_comp = expoutput.TiaraFCOutput(
            ["32c", "31c"], conf, session_mock, multiplerun=True
        )
        self.benchoutput_comp.compare()
        assert True

    def test_fngoutput(self, session_mock: MockUpSession):

        config = session_mock.conf.exp_default.set_index("Description")
        conf = config.iloc[1]
        os.makedirs(session_mock.path_comparison)
        os.makedirs(session_mock.path_single)
        self.benchoutput_comp = expoutput.FNGOutput(
            ["99c", "98c"], conf, session_mock, multiplerun=True
        )
        self.benchoutput_comp.compare()
        assert True
