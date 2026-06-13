from copy import deepcopy

from .average import AveragesPipeline
from .component import ComponentsPipeline
from .epoch import EpochPipeline
from .input import InputPipeline
from .preproc import PreprocPipeline


class ParticipantPipeline:
    """The participant pipeline for processing the EEG data of a single
    participant."""

    def __init__(
        self,
        input_pipeline: InputPipeline,
        preproc_pipeline: PreprocPipeline,
        epoch_pipeline: EpochPipeline,
        components_pipeline: ComponentsPipeline,
        averages_pipeline: AveragesPipeline,
    ):
        self.input_pipeline = input_pipeline
        self.preproc_pipeline = preproc_pipeline
        self.epoch_pipeline = epoch_pipeline
        self.components_pipeline = components_pipeline
        self.averages_pipeline = averages_pipeline

    def copy(self):
        """Create a copy of the ParticipantPipeline instance."""

        return deepcopy(self)

    def run(self):
        self.input_pipeline.run()
        self.raw = self.input_pipeline.raw
        self.log = self.input_pipeline.log
        self.besa = self.input_pipeline.besa

        self.preproc_pipeline.run(self.raw, self.besa)
        self.raw_preproc = self.preproc_pipeline.raw

        self.epoch_pipeline.run(self.raw_preproc, self.log)
        self.epochs = self.epoch_pipeline.epochs
        self.bad_ixs = self.epoch_pipeline.bad_ixs

        # TODO: Maybe this could be done on the continuous raw data (i.e.,
        # fully within the preproc pipeline) rather then on the epochs.
        # Let's check once we added automatic break detection (#212).
        if self.preproc_pipeline.bad_channels == "auto":
            self._detect_bad_channels_and_rerun()

        self.components_pipeline.run(self.epochs, self.bad_ixs)

        self.averages_pipeline.run(self.epochs, self.bad_ixs)

    def _detect_bad_channels_and_rerun(self):
        bad_channels = self.epoch_pipeline.detect_bad_channels()

        if len(bad_channels) > 0:
            self.preproc_pipeline_rerun = self.preproc_pipeline.copy()
            self.preproc_pipeline_rerun.bad_channels = bad_channels

            self.preproc_pipeline_rerun.run(self.raw, self.besa)
            self.raw_preproc_rerun = self.preproc_pipeline_rerun.raw
            self.epoch_pipeline.run(self.raw_preproc_rerun, self.log)
