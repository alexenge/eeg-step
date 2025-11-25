import numpy as np
from mne import Epochs

from step.component import ComponentConfig, ComponentPipeline


def test_component_config(sample_component_config_n2):
    """Test the ComponentConfig class."""

    config = sample_component_config_n2

    assert isinstance(config, ComponentConfig)
    assert isinstance(config.name, str)
    assert isinstance(config.tmin, float)
    assert isinstance(config.tmax, float)
    assert isinstance(config.roi, list)
    assert all(isinstance(chan, str) for chan in config.roi)
    assert isinstance(config.compute_se, bool)


def test_component_pipeline(sample_component_pipeline):
    """Test the ComponentPipeline class."""

    pipeline = sample_component_pipeline

    n_epochs = len(pipeline.epochs)

    assert isinstance(pipeline, ComponentPipeline)
    assert isinstance(pipeline.epochs, Epochs)
    assert isinstance(pipeline.bad_ixs, list)
    assert isinstance(pipeline.roi, list)
    assert isinstance(pipeline.data, np.ndarray)
    assert isinstance(pipeline.amplitudes, np.ndarray)
    assert pipeline.amplitudes.shape == (n_epochs,)
    assert pipeline.standard_deviations.shape == (n_epochs,)
    assert pipeline.standard_errors.shape == (n_epochs,)
    assert "N2" in pipeline.epochs.metadata.columns
    assert "N2_se" in pipeline.epochs.metadata.columns
    # Some bad epochs were rejected
    assert pipeline.epochs.metadata["N2"].isna().sum() > 0
    assert (
        pipeline.epochs.metadata["N2_se"].isna().sum()
        == pipeline.epochs.metadata["N2"].isna().sum()
    )
