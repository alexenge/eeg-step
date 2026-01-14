from os import PathLike
import pandas as pd

from .input import InputConfig
from .participant import ParticipantConfig, ParticipantPipeline


class GroupPipeline:
    """The group pipeline for processing the EEG data at the group level"""

    def __init__(
        self,
        raw_files: list[str | PathLike] | str | PathLike,
        log_files: list[str | PathLike | pd.DataFrame] | str | PathLike = None,
        besa_files: list[str | PathLike] | str | PathLike = None,
    ):
        for raw_file, log_file, besa_file in zip(raw_files, log_files, besa_files):
            self.input_config = InputConfig(
                raw_file=raw_file, log_file=log_file, besa_file=besa_file
            )
            self.participant_config = ParticipantConfig(self.input_config)
            self.participant_pipeline = ParticipantPipeline(self.participant_config)

    def run(self):
        self.participant_pipeline.run()
