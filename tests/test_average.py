from mne import Evoked

from step.average import AverageConfig, AveragePipeline
import numpy as np


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

    # TODO: Take channels + time window from component definition
    mean_blurr = evoked_blurr.pick_channels(["Pz"]).crop(tmin=0.4, tmax=0.5).data.mean()
    mean_normal = (
        evoked_normal.pick_channels(["Pz"]).crop(tmin=0.4, tmax=0.5).data.mean()
    )
    assert (mean_normal - mean_blurr) > np.float64(1.0e-06)  # Expeced P3 effect
