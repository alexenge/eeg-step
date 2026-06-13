import pandas as pd
from mne import Epochs
from mne.io import BaseRaw

from step.average import AveragesPipeline
from step.component import ComponentsPipeline
from step.epoch import EpochPipeline
from step.input import InputPipeline
from step.participant import ParticipantPipeline
from step.preproc import PreprocPipeline


def test_participant_pipeline(sample_participant_pipeline):
    """Tests the ParticipantPipeline class."""

    pipeline = sample_participant_pipeline
    assert isinstance(pipeline, ParticipantPipeline)

    assert isinstance(pipeline.input_pipeline, InputPipeline)
    assert isinstance(pipeline.preproc_pipeline, PreprocPipeline)
    assert isinstance(pipeline.epoch_pipeline, EpochPipeline)
    assert isinstance(pipeline.components_pipeline, ComponentsPipeline)
    assert isinstance(pipeline.averages_pipeline, AveragesPipeline)

    assert isinstance(pipeline.raw, BaseRaw)
    assert isinstance(pipeline.log, pd.DataFrame)
    assert isinstance(pipeline.besa, pd.DataFrame)
    assert isinstance(pipeline.raw_preproc, BaseRaw)
    assert isinstance(pipeline.epochs, Epochs)
    assert isinstance(pipeline.bad_ixs, list)
