import numpy as np
from mne import Evoked

from step.average import Average, AveragePipeline, AveragesPipeline


def test_average(sample_average_blurr):
    """Test the Average class."""

    average = sample_average_blurr

    assert isinstance(average, Average)
    assert isinstance(average.name, str)
    assert isinstance(average.query, str)


def test_average_pipeline(sample_average_pipeline):
    """Test the AveragePipeline class."""

    pipeline = sample_average_pipeline

    assert isinstance(pipeline, AveragePipeline)

    assert isinstance(pipeline.average, Average)

    assert isinstance(pipeline.evoked, Evoked)
    assert pipeline.evoked.comment == pipeline.average.name


def test_averages_pipeline(sample_averages_pipeline):
    """Test multiple AveragePipeline instances."""

    pipeline = sample_averages_pipeline

    assert isinstance(pipeline, AveragesPipeline)

    assert isinstance(pipeline.averages, list)
    assert isinstance(pipeline.averages_, list)

    averages = sample_averages_pipeline.averages
    assert all(isinstance(average, Average) for average in averages)

    average_pipelines = sample_averages_pipeline.average_pipelines
    assert isinstance(average_pipelines, dict)
    assert set(average_pipelines.keys()) == {average.name for average in averages}
    assert all(isinstance(p, AveragePipeline) for p in average_pipelines.values())
    assert len(average_pipelines) == len(averages)

    # Expeced P3b effect for normal vs. blurr
    evoked_blurr = average_pipelines["blurr"].evoked
    evoked_normal = average_pipelines["normal"].evoked
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
    assert (mean_normal - mean_blurr) > np.float64(1.0e-06)
