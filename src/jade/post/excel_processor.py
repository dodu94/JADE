from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd

from jade.config.pp_config import ConfigExcelProcessor
from jade.helper.aux_functions import PathLike, print_code_lib
from jade.post.excel_routines import TableFactory

TITLE = "{}-{} Vs {}-{}. Result: {}"
FILE_NAME = "{}_{}-{}_Vs_{}-{}.xlsx"


class ExcelProcessor:
    def __init__(
        self,
        raw_root: PathLike,
        excel_folder_path: PathLike,
        cfg: ConfigExcelProcessor,
    ) -> None:
        self.excel_folder_path = excel_folder_path
        self.raw_root = raw_root
        self.cfg = cfg

    def process(self) -> None:
        for i, (code, lib) in enumerate(self.cfg.libcodes):
            codelib = print_code_lib(code, lib)
            logging.info("Parsing reference data")
            raw_folder = Path(self.raw_root, codelib, self.cfg.benchmark)

            # First store all reference dfs
            reference_dfs = {}
            if i == 0:
                ref_code = code
                ref_lib = lib
                for table_cfg in self.cfg.tables:
                    target_df = self._get_table_df(table_cfg.results, raw_folder)
                    reference_dfs[table_cfg.name] = target_df

            # then we can produce one excel comparison file for each target
            else:
                outfile = Path(
                    self.excel_folder_path,
                    FILE_NAME.format(self.cfg.benchmark, ref_code, ref_lib, code, lib),
                )
                logging.info(f"Writing the resulting excel file {outfile}")
                with pd.ExcelWriter(self.excel_folder_path) as writer:
                    for table_cfg in self.cfg.tables:
                        # this get a concatenated dataframe with all results that needs to be
                        # in the table
                        target_df = self._get_table_df(table_cfg.results, raw_folder)
                        title = TITLE.format(
                            ref_code.value, ref_lib, code.value, lib, table_cfg.name
                        )
                        ref_df = reference_dfs[table_cfg.name]
                        table = TableFactory.create_table(
                            table_cfg.table_type,
                            [title, writer, ref_df, target_df, table_cfg],
                        )
                        table.add_sheets()

    @staticmethod
    def _get_table_df(results: list[int | str], raw_folder: PathLike) -> pd.DataFrame:
        """given a list of results, get the concatenated dataframe"""
        dfs = []
        for result in results:
            # this gets a concatenated dataframe for each result for different runs
            df = ExcelProcessor._get_concat_df_results(result, raw_folder)
            df["Result"] = result
            dfs.append(df)
        return pd.concat(dfs)

    @staticmethod
    def _get_concat_df_results(
        target_result: int | str, folder: PathLike
    ) -> pd.DataFrame:
        """given a result ID, locate, read the dataframes and concat them (from different
        single runs)"""
        dfs = []
        for file in os.listdir(folder):
            result, run_name = file.split(" ")
            if result == target_result:
                df = pd.read_csv(Path(folder, file))
                df["Case"] = run_name
                dfs.append(df)
        if len(dfs) == 0:
            logging.warning(f"No data found for {target_result}")
            return pd.DataFrame()
        return pd.concat(dfs)
