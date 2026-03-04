from os import PathLike

import pandas as pd

from .input import InputConfig
from .participant import ParticipantConfig, ParticipantPipeline
from .preproc import PreprocConfig


class GroupPipeline:
    """The group pipeline for processing the EEG data at the group level"""

    def __init__(
        self,
        raw_files: list[str | PathLike] | str | PathLike,
        log_files: list[str | PathLike | pd.DataFrame] | str | PathLike = None,
        besa_files: list[str | PathLike] | str | PathLike = None,
        downsample_sfreq: float = None,
        heog_channels: list[str] | str = "auto",
        veog_channels: list[str] | str = "auto",
        montage: str | PathLike = "easycap-M1",
        bad_channels: dict[list[str]] | str = "auto",
        ref_channels: list[str] | str = "average",
        ica_method: str = "fastica",
        ica_n_components: int | float = None,
        ica_eog_channels: list[str] | str = "auto",
        highpass_freq: float = 0.1,
        lowpass_freq: float = 40.0,
    ):
        self.participant_pipelines = dict()
        for raw_file, log_file, besa_file in zip(raw_files, log_files, besa_files):
            input_config = InputConfig(
                raw_file=raw_file, log_file=log_file, besa_file=besa_file
            )

            preproc_config = PreprocConfig(
                downsample_sfreq=downsample_sfreq,
                heog_channels=heog_channels,
                veog_channels=veog_channels,
                montage=montage,
                bad_channels=bad_channels,
                ref_channels=ref_channels,
                ica_method=ica_method,
                ica_n_components=ica_n_components,
                ica_eog_channels=ica_eog_channels,
                highpass_freq=highpass_freq,
                lowpass_freq=lowpass_freq,
            )

            participant_config = ParticipantConfig(input_config, preproc_config)
            participant_pipeline = ParticipantPipeline(participant_config)
            participant_id = participant_pipeline.input_pipeline.participant_id
            self.participant_pipelines[participant_id] = participant_pipeline

    def run(self):
        for participant_pipeline in self.participant_pipelines.values():
            participant_pipeline.run()

    def _process_bad_channels(self):
        n_participants = len(self.raw_files)

        if self.bad_channels is list:
            self.bad_channels_ = self.bad_channels
        elif self.bad_channels == "auto":
            self.bad_channels_ = ["auto"] * n_participants
        elif self.bad_channels is dict:
            print("Oh nooooo!")

    def _get_participant_ids(self):
        return True
