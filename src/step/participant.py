from dataclasses import dataclass

from .average import AverageConfig, AveragePipeline
from .component import ComponentsPipeline
from .epoch import EpochPipeline
from .input import InputPipeline
from .preproc import PreprocPipeline


@dataclass
class ParticipantConfig:
    """The configuration for the participant pipeline."""

    average_configs: list[AverageConfig] = None


class ParticipantPipeline:
    """The participant pipeline for processing the EEG data of a single
    participant."""

    def __init__(
        self,
        config: ParticipantConfig,
        input_pipeline: InputPipeline,
        preproc_pipeline: PreprocPipeline,
        epoch_pipeline: EpochPipeline,
        components_pipeline: ComponentsPipeline,
    ):
        self.input_pipeline = input_pipeline
        self.preproc_pipeline = preproc_pipeline
        self.epoch_pipeline = epoch_pipeline
        self.components_pipeline = components_pipeline
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
        if self.preproc_pipeline.bad_channels == "auto":
            self._detect_bad_channels_and_rerun()

        self.components_pipeline.run(
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
            self.preproc_pipeline.bad_channels = bad_channels
            self.preproc_pipeline.run(self.input_pipeline.raw, self.input_pipeline.besa)
            self.epoch_pipeline.run(self.preproc_pipeline.raw, self.input_pipeline.log)
