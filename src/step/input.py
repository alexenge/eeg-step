from os import PathLike
from pathlib import Path

import chardet
import pandas as pd
from mne.io import concatenate_raws, read_raw
from pandas.api.types import is_list_like


class InputPipeline:
    """The input pipeline for reading the raw EEG data and associated files."""

    def __init__(
        self,
        participant_id: str,
        raw_file: str | PathLike | list[str | PathLike],
        log_file: str | PathLike | pd.DataFrame = None,
        besa_file: str | PathLike = None,
    ):
        self.participant_id = participant_id
        self.raw_file = raw_file
        self.log_file = log_file
        self.besa_file = besa_file

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

        if is_list_like(self.raw_file):
            raws = [read_raw(elem, preload=True) for elem in self.raw_file]
            self.raw = concatenate_raws(raws)

        else:
            self.raw = read_raw(self.raw_file, preload=True)

    def _add_participant_id_to_raw(self):
        """Add the participant ID to the raw data's info dictionary."""

        if self.raw.info["subject_info"] is not None:
            self.raw.info["subject_info"].update({"his_id": self.participant_id})
        else:
            self.raw.info["subject_info"] = {"his_id": self.participant_id}

    def _read_log(self):
        """Read the behavioral log file with information about each EEG
        trial."""

        if self.log_file is None:
            self.log = None

        elif isinstance(self.log_file, pd.DataFrame):
            self.log = self.log_file

        else:
            with open(self.log_file, "rb") as f:
                data = f.read()
            chardet_res = chardet.detect(data)
            encoding = chardet_res["encoding"]

            if Path(self.log_file).suffix == ".csv":
                self.log = pd.read_csv(self.log_file, encoding=encoding)

            else:
                self.log = pd.read_csv(self.log_file, delimiter="\t", encoding=encoding)

    def _add_participant_id_to_log(self):
        """Add the participant ID to the log as a new column."""

        self.log.insert(0, column="participant_id", value=self.participant_id)

    def _read_besa(self):
        """Read the BESA file containing the ocular correction matrix."""

        if self.besa_file is None:
            self.besa = None

        else:
            self.besa = pd.read_csv(self.besa_file, delimiter="\t", index_col=0)
