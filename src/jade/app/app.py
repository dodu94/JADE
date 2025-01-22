from __future__ import annotations

import logging
import os
import shutil
import time
from importlib.resources import files
from pathlib import Path

import yaml
from tqdm import tqdm

import jade.resources as res
from jade.app.fetch import fetch_iaea_inputs
from jade.config.paths_tree import PathsTree
from jade.config.pp_config import PostProcessConfig
from jade.config.raw_config import ConfigRawProcessor
from jade.config.run_config import RunConfig
from jade.config.status import GlobalStatus
from jade.gui.config_gui import ConfigGUI
from jade.helper.aux_functions import PathLike, get_code_lib, print_code_lib
from jade.helper.constants import CODE, FIRST_INITIALIZATION, JADE_TITLE
from jade.post.excel_processor import ExcelProcessor
from jade.post.raw_processor import RawProcessor
from jade.run.benchmark import BenchmarkRunFactory

DEFAULT_SETTINGS_PATH = files(res).joinpath("default_cfg")


class JadeApp:
    def __init__(self, root: PathLike | None = None, skip_init: bool = False):
        if root is None:
            root = os.getcwd()

        # Initialize the local installation if it was never done
        self.tree = PathsTree(root)
        if len(os.listdir(root)) == 0 and not skip_init:
            self.tree.init_tree()
            self.restore_default_cfg(FIRST_INITIALIZATION)
            self.update_inputs()
            return

        # parse the config files
        self.run_cfg = RunConfig.from_root(self.tree)
        # TODO read here also the post-processing config

        # Compute the global status
        self.status = GlobalStatus(
            simulations_path=self.tree.simulations,
            raw_results_path=self.tree.raw,
        )

    def initialize_log(self) -> None:
        """Initialize the custom python logger for JADE."""
        log = os.path.join(
            self.tree.logs, "Log " + time.ctime().replace(":", "-") + ".txt"
        )
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)

        # set the logging to a file and keep warnings to video
        # Create a file handler for logging INFO level messages
        file_handler = logging.FileHandler(log, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        # Create a console handler for logging WARNING and ERROR level messages
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)

        for handler in logger.handlers:
            # there should already be a streamhandler
            if isinstance(handler, logging.StreamHandler):
                handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        logging.debug(JADE_TITLE)

    def update_inputs(self):
        """Update the benchmark inputs for the simulations.

        This will re-fetch inputs from the various repositories that feed JADE.
        """
        fetch_iaea_inputs(self.tree.benchmark_input_templates, self.tree.exp_data)

    def restore_default_cfg(self, msg: str = ""):
        """Reset the configuration files to installation default. The session
        is re-initialized.
        """
        # Remove the user configurations
        shutil.rmtree(self.tree.cfg.path)
        # Copy default configurations
        shutil.copytree(DEFAULT_SETTINGS_PATH, self.tree.cfg.path)
        print(msg)

    def rmv_runtpe(self):
        """Remove the runtpe files from the simulation folders of mcnp."""
        # Walk through the root folder and remove the runtpe files
        for pathroot, folder, files in os.walk(self.tree.simulations):
            if len(files) > 0 and len(folder) == 0:  # only files and no folders
                # get the code
                codelib = Path(pathroot).parent.parent.name
                code, _ = get_code_lib(codelib)
                if CODE(code) == CODE.MCNP:  # only for mcnp
                    for file in os.listdir(pathroot):
                        if file.endswith(".r"):
                            os.remove(os.path.join(pathroot, file))
        logging.info("Runtpe files were removed successfully")

    def run_benchmarks(self):
        """Run the benchmarks according to the configuration."""
        logging.info("Running benchmarks")
        # first thing do to is to check if the benchmarks were already run
        simulated = []
        for bench_name, cfg in self.run_cfg.benchmarks.items():
            for code, lib in cfg.run:
                # First of all check if the benchmark was already run for some of the
                # benchmarks
                if self.status.was_simulated(code, lib.name, bench_name):
                    simulated.append((code, lib, bench_name))

        # if yes, ask for confirmation before overriding
        proceed_flag = True
        if len(simulated) > 0:
            logging.warning("The following benchmarks were already simulated:")
            for code, lib, bench_name in simulated:
                logging.warning(f"{code} - {lib.name}: {bench_name}")
            logging.warning("If you continue, the results will be overwritten.")
            proceed_flag = input("Do you want to continue? [y/n]: ").lower() == "y"

        if proceed_flag:
            for bench_name, cfg in self.run_cfg.benchmarks.items():
                benchmark = BenchmarkRunFactory.create(
                    cfg,
                    self.tree.simulations,
                    self.tree.benchmark_input_templates,
                    self.run_cfg.env_vars,
                )
                benchmark.run()
        logging.info("Benchmarks run completed.")

    def raw_process(self):
        """Process the raw data from the simulations."""
        logging.info("Processing raw data")
        # first identify all simulations that were successful but were not processed
        root_cfg = self.tree.cfg.bench_raw
        successful = self.status.get_successful_simulations()
        to_process = {}
        for code, lib, bench in successful:
            if (code, lib, bench) not in self.status.raw_data:
                # get the correspondent raw processor configuration
                cfg_file = Path(root_cfg, f"{code.value}/{bench}.yaml")
                raw_cfg = ConfigRawProcessor.from_yaml(cfg_file)
                to_process[(code, lib, bench)] = raw_cfg
                logging.info(f"Processing {code.value} {lib} {bench} benchmarks")

        # process the raw data
        for (code, lib, bench), cfg in tqdm(to_process.items(), desc="Process raw"):
            folders = self.tree.get_bench_sim_folders(code, lib, bench)
            out_folder = Path(self.tree.raw, print_code_lib(code, lib), bench)
            # always ovveride eventual previous results here
            if out_folder.exists():
                shutil.rmtree(out_folder)
            os.makedirs(out_folder, exist_ok=True)
            for sim_folder, _ in folders:
                processor = RawProcessor(cfg, sim_folder, out_folder)
                processor.process_raw_data()

        logging.info("Raw data processing completed.")

    def post_process(self):
        """Post-process the data."""
        logging.info("Post-processing data")
        # recover the post-processing benchmark configurations
        bench_cfg = PostProcessConfig(self.tree.cfg.bench_pp)
        # load the pp code-lib requests
        with open(self.tree.cfg.pp_cfg) as f:
            pp_cfg = yaml.safe_load(f)
        for benchmark, options in pp_cfg.items():
            if options["process"]:
                # prepare the new paths
                pp_path = self.tree.get_new_post_bench_path(benchmark)
                excel_folder = Path(pp_path, "excel")
                os.mkdir(excel_folder)
                # perform the excel processing
                excel_cfg = bench_cfg.excel_cfgs[benchmark]
                options["codelibs"]
                processor = ExcelProcessor(
                    self.tree.raw,
                    excel_folder,
                    excel_cfg,
                    options["codelibs"],
                )
                processor.process()

    def start_config_gui(self):
        """Start the configuration GUI."""
        logging.info("Starting the configuration GUI")
        app = ConfigGUI(self.tree.cfg.run_cfg, self.tree.cfg.libs_cfg)
        app.window.mainloop()