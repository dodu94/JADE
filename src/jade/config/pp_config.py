from __future__ import annotations

import os
from pathlib import Path

from jade.config.excel_config import ConfigExcelProcessor
from jade.helper.aux_functions import PathLike


class PostProcessConfig:
    def __init__(self, root_cfg_pp: PathLike):
        # get all available config excel processors
        excel_cfgs = {}
        for file in os.listdir(Path(root_cfg_pp, "excel")):
            if file.endswith(".yaml") or file.endswith(".yml"):
                cfg = ConfigExcelProcessor.from_yaml(Path(root_cfg_pp, file))
                excel_cfgs[cfg.benchmark] = cfg
        self.excel_cfgs = excel_cfgs
        # TODO get all available config atlas processors