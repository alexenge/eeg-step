from dataclasses import dataclass
from os import PathLike
from pathlib import Path

import chardet
import pandas as pd
from mne.io import concatenate_raws, read_raw
from pandas.api.types import is_list_like


@dataclass
class InputConfig:
    """The configuration for the input pipeline."""

    participant_id: str
    raw_file: str | PathLike | list[str | PathLike]
    log_file: str | PathLike | pd.DataFrame = None
    besa_file: str | PathLike = None


class InputPipeline:
    """The input pipeline for reading the raw EEG data and associated files."""

    def __init__(self, config):
        assert isinstance(config, InputConfig), (
            "`config` must be an instance of the `InputConfig` class"
        )
        self.config = config

    def run(self):
        """Run the input pipeline."""

        self._read_raw()
        self._add_participant_id_to_raw()

        self._read_log()
        if self.log is not None:
            self._add_participant_id_to_log()

        self._read_besa()

    def _read_raw(self):
        """Read EEG raw data from the specified file(s)."""

        if is_list_like(self.config.raw_file):
            raws = [read_raw(elem, preload=True) for elem in self.config.raw_file]
            self.raw = concatenate_raws(raws)

        else:
            self.raw = read_raw(self.config.raw_file, preload=True)

    def _add_participant_id_to_raw(self):
        """Add the participant ID to the raw data's info dictionary."""

        if self.raw.info["subject_info"] is not None:
            self.raw.info["subject_info"].update({"his_id": self.config.participant_id})
        else:
            self.raw.info["subject_info"] = {"his_id": self.config.participant_id}

    def _read_log(self):
        """Read the behavioral log file with information about each EEG
        trial."""

        if self.config.log_file is None:
            self.log = None

        elif isinstance(self.config.log_file, pd.DataFrame):
            self.log = self.config.log_file

        else:
            with open(self.config.log_file, "rb") as f:
                data = f.read()
            chardet_res = chardet.detect(data)
            encoding = chardet_res["encoding"]

            if Path(self.config.log_file).suffix == ".csv":
                self.log = pd.read_csv(self.config.log_file, encoding=encoding)

            else:
                self.log = pd.read_csv(
                    self.config.log_file, delimiter="\t", encoding=encoding
                )

    def _add_participant_id_to_log(self):
        """Add the participant ID to the log as a new column."""

        self.log.insert(0, column="participant_id", value=self.config.participant_id)

    def _read_besa(self):
        """Read the BESA file containing the ocular correction matrix."""

        if self.config.besa_file is None:
            self.besa = None

        else:
            self.besa = pd.read_csv(self.config.besa_file, delimiter="\t", index_col=0)
