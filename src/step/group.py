from os import PathLike

import pandas as pd

from .component import ComponentConfig
from .epoch import EpochConfig
from .helpers import (
    _dict_to_average_configs,
    _dict_to_list,
    _get_participant_id,
    _process_files_input,
)
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
        triggers: list[int] = None,
        triggers_column: str = None,
        tmin: float = -0.2,
        tmax: float = 0.8,
        baseline: tuple[float, float] = (-0.2, 0.0),
        reject: float = 200.0,
        components: list[ComponentConfig] = None,
        average_by: dict = None,
    ):
        self.raw_files = raw_files
        self.log_files = log_files
        self.besa_files = besa_files
        self.downsample_sfreq = downsample_sfreq
        self.heog_channels = heog_channels
        self.veog_channels = veog_channels
        self.montage = montage
        self.bad_channels = bad_channels
        self.ref_channels = ref_channels
        self.ica_method = ica_method
        self.ica_n_components = ica_n_components
        self.ica_eog_channels = ica_eog_channels
        self.highpass_freq = highpass_freq
        self.lowpass_freq = lowpass_freq
        self.triggers = triggers
        self.triggers_column = triggers_column
        self.tmin = tmin
        self.tmax = tmax
        self.baseline = baseline
        self.reject = reject
        self.components = components
        self.average_by = average_by

        self._process_raw_files()
        self._get_participant_ids()
        self._process_log_files()
        self._process_besa_files()
        self._process_bad_channels()

        # Common configurations for all participants
        epoch_config = EpochConfig(
            triggers=triggers,
            triggers_column=triggers_column,
            tmin=tmin,
            tmax=tmax,
            baseline=baseline,
            reject=reject,
        )
        component_configs = components
        average_configs = _dict_to_average_configs(average_by)

        # Participant-specific configurations and pipelines
        self.participant_pipelines = dict()
        for (
            raw_file,
            log_file,
            besa_file,
            participant_id,
            participant_bad_channels,
        ) in zip(
            self.raw_files_,
            self.log_files_,
            self.besa_files_,
            self.participant_ids,
            self.bad_channels_,
        ):
            input_config = InputConfig(
                raw_file=raw_file,
                log_file=log_file,
                besa_file=besa_file,
                participant_id=participant_id,
            )

            preproc_config = PreprocConfig(
                downsample_sfreq=downsample_sfreq,
                heog_channels=heog_channels,
                veog_channels=veog_channels,
                montage=montage,
                bad_channels=participant_bad_channels,
                ref_channels=ref_channels,
                ica_method=ica_method,
                ica_n_components=ica_n_components,
                ica_eog_channels=ica_eog_channels,
                highpass_freq=highpass_freq,
                lowpass_freq=lowpass_freq,
            )

            participant_config = ParticipantConfig(
                input_config,
                preproc_config,
                epoch_config,
                component_configs,
                average_configs,
            )

            participant_pipeline = ParticipantPipeline(participant_config)
            self.participant_pipelines[participant_id] = participant_pipeline

    def run(self):
        for participant_pipeline in self.participant_pipelines.values():
            participant_pipeline.run()

    def _process_raw_files(self):
        self.raw_files_ = _process_files_input(self.raw_files, file_extensions=["vhdr"])

    def _process_log_files(self):
        self.log_files_ = _process_files_input(
            self.log_files,
            file_extensions=["csv", "tsv", "txt"],
            n_out=len(self.participant_ids),
        )

    def _process_besa_files(self):
        self.besa_files_ = _process_files_input(
            self.besa_files,
            file_extensions=["matrix"],
            n_out=len(self.participant_ids),
        )

    def _get_participant_ids(self):
        self.participant_ids = [
            _get_participant_id(raw_file) for raw_file in self.raw_files_
        ]

    def _process_bad_channels(self):
        if isinstance(self.bad_channels, list):
            self.bad_channels_ = self.bad_channels
        elif self.bad_channels == "auto":
            self.bad_channels_ = ["auto"] * len(self.participant_ids)
        elif isinstance(self.bad_channels, dict):
            self.bad_channels_ = _dict_to_list(self.bad_channels, self.participant_ids)
        else:
            raise ValueError(
                "`bad_channels` must be a list of lists, a dictionary, or 'auto'"
            )
