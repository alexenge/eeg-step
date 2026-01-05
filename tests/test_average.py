import numpy as np
from mne import Evoked

from step.average import AverageConfig, AveragePipeline


def test_average_config(sample_average_config_blurr):
    """Test the AverageConfig class."""

    config = sample_average_config_blurr

    assert isinstance(config, AverageConfig)
    assert isinstance(config.name, str)
    assert isinstance(config.query, str)


def test_average_pipeline_blurr(sample_average_pipeline_blurr):
    """Test the AveragePipeline class."""

    pipeline = sample_average_pipeline_blurr

    assert isinstance(pipeline, AveragePipeline)
    assert isinstance(pipeline.evoked, Evoked)
    assert pipeline.evoked.comment == pipeline.config.name


def test_average_pipelines(
    sample_average_pipeline_blurr, sample_average_pipeline_normal
):
    """Test multiple AveragePipeline instances."""

    evoked_blurr = sample_average_pipeline_blurr.evoked
    evoked_normal = sample_average_pipeline_normal.evoked

    assert evoked_blurr.comment != evoked_normal.comment
    assert evoked_blurr.data.shape == evoked_normal.data.shape

    p3b_channels = [
        "CP3",
        "CP1",
        "CPz",
        "CP2",
        "CP4",
        "P3",
        "Pz",
        "P4",
        "PO3",
        "POz",
        "PO4",
    ]
    p3b_tmin = 0.4
    p3b_tmax = 0.55

    mean_blurr = (
        evoked_blurr.pick_channels(p3b_channels)
        .crop(tmin=p3b_tmin, tmax=p3b_tmax)
        .data.mean()
    )
    mean_normal = (
        evoked_normal.pick_channels(p3b_channels)
        .crop(tmin=p3b_tmin, tmax=p3b_tmax)
        .data.mean()
    )

    assert (mean_normal - mean_blurr) > np.float64(1.0e-06)  # Expeced P3b effect
