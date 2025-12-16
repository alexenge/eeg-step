from dataclasses import dataclass

from .component import ComponentConfig, ComponentPipeline
from .epoch import EpochConfig, EpochPipeline
from .input import InputConfig, InputPipeline
from .preproc import PreprocConfig, PreprocPipeline
from .average import AverageConfig, AveragePipeline


@dataclass
class ParticipantConfig:
    """The configuration for the participant pipeline."""

    input_config: InputConfig = None
    preproc_config: PreprocConfig = None
    epoch_config: EpochConfig = None
    component_configs: list[ComponentConfig] = None
    average_configs: list[AverageConfig] = None


class ParticipantPipeline:
    """The participant pipeline for processing the EEG data of a single
    participant."""

    def __init__(self, config: ParticipantConfig):
        self.input_pipeline = InputPipeline(config.input_config)
        self.preproc_pipeline = PreprocPipeline(config.preproc_config)
        self.epoch_pipeline = EpochPipeline(config.epoch_config)
        self.component_pipelines = [
            ComponentPipeline(cfg) for cfg in config.component_configs
        ]
        self.average_pipelines = [
            AveragePipeline(cfg) for cfg in config.average_configs
        ]

    def run(self):
        self.input_pipeline.run()

        self.preproc_pipeline.run(self.input_pipeline.raw, self.input_pipeline.besa)

        self.epoch_pipeline.run(self.preproc_pipeline.raw, self.input_pipeline.log)

        # TODO: Maybe this could be done on the continuous raw data (i.e.,
        # fully within the preproc pipeline) rather then on the epochs.
        # Let's check once we added automatic break detection (#212).
        if self.preproc_pipeline.config.bad_channels == "auto":
            self._detect_bad_channels_and_rerun()

        for component_pipeline in self.component_pipelines:
            component_pipeline.run(
                self.epoch_pipeline.epochs, self.epoch_pipeline.bad_ixs
            )

        for average_pipeline in self.average_pipelines:
            average_pipeline.run(
                self.epoch_pipeline.epochs,
                self.epoch_pipeline.bad_ixs,
            )

    def _detect_bad_channels_and_rerun(self):
        bad_channels = self.epoch_pipeline.detect_bad_channels()

        if len(bad_channels) > 0:
            self.preproc_pipeline.config.bad_channels = bad_channels
            self.preproc_pipeline.run(self.input_pipeline.raw, self.input_pipeline.besa)
            self.epoch_pipeline.run(self.preproc_pipeline.raw, self.input_pipeline.log)
