import numpy as np
import pandas as pd
from mne import Epochs

from step.epoch import EpochPipeline


def test_epoch_pipeline(sample_epoch_pipeline):
    """Tests the EpochPipeline class."""

    pipeline = sample_epoch_pipeline

    assert isinstance(pipeline, EpochPipeline)

    assert pipeline.triggers == [
        201,
        202,
        203,
        204,
        205,
        206,
        207,
        208,
        211,
        212,
        213,
        214,
        215,
        216,
        217,
        218,
    ]
    assert isinstance(pipeline.events, np.ndarray)
    assert all(trigger in pipeline.events[:, 2] for trigger in pipeline.triggers)

    assert pipeline.triggers_column is None

    assert isinstance(pipeline.event_id, dict)
    assert set(pipeline.event_id.values()) == set(pipeline.triggers)
    assert set(pipeline.event_id.keys()) == set(map(str, pipeline.triggers))

    assert pipeline.tmin == -0.2
    assert pipeline.tmax == 0.8
    assert pipeline.baseline == (-0.2, 0.0)
    assert isinstance(pipeline.epochs, Epochs)
    assert isinstance(pipeline.epochs.metadata, pd.DataFrame)

    assert pipeline.reject == 200.0
    assert isinstance(pipeline.bad_ixs, list)
    assert len(pipeline.bad_ixs) > 0


def test_epoch_pipeline_match(sample_epoch_pipeline_match):
    """Tests the EpochPipeline class with automatic matchin to missing EEG trials."""

    pipeline = sample_epoch_pipeline_match

    assert len(pipeline.epochs) > 0
    assert len(pipeline.epochs) < 1920
    assert np.array_equal(
        pipeline.epochs.events[:, 2],
        pipeline.epochs.metadata[pipeline.triggers_column].values,
    )
