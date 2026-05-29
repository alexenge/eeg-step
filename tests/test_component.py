import numpy as np
from mne import Epochs

from step.component import Component, ComponentPipeline, ComponentsPipeline


def test_component(sample_component_n2):
    """Test the Component class."""

    component = sample_component_n2

    assert isinstance(component, Component)
    assert isinstance(component.name, str)
    assert isinstance(component.tmin, float)
    assert isinstance(component.tmax, float)
    assert isinstance(component.roi, list)
    assert all(isinstance(chan, str) for chan in component.roi)


def test_component_pipeline(sample_component_pipeline):
    """Test the ComponentPipeline class."""

    pipeline = sample_component_pipeline

    assert isinstance(pipeline, ComponentPipeline)

    assert isinstance(pipeline.component, Component)
    assert isinstance(pipeline.compute_se, bool)

    assert isinstance(pipeline.epochs, Epochs)
    assert isinstance(pipeline.bad_ixs, list)

    assert isinstance(pipeline.roi, list)
    assert isinstance(pipeline.data, np.ndarray)
    assert isinstance(pipeline.amplitudes, np.ndarray)

    n_epochs = len(pipeline.epochs)
    assert pipeline.amplitudes.shape == (n_epochs,)
    assert "N2" in pipeline.epochs.metadata.columns

    assert pipeline.standard_deviations.shape == (n_epochs,)
    assert pipeline.standard_errors.shape == (n_epochs,)
    assert "N2_se" in pipeline.epochs.metadata.columns

    # Some bad epochs were rejected
    assert pipeline.epochs.metadata["N2"].isna().sum() > 0
    assert (
        pipeline.epochs.metadata["N2_se"].isna().sum()
        == pipeline.epochs.metadata["N2"].isna().sum()
    )


def test_components_pipeline(sample_components_pipeline):
    """Test the ComponentsPipeline class."""

    pipeline = sample_components_pipeline

    assert isinstance(pipeline, ComponentsPipeline)

    assert isinstance(pipeline.components, list)
    assert all(isinstance(component, Component) for component in pipeline.components)

    assert isinstance(pipeline.compute_se, bool)

    assert isinstance(pipeline.component_pipelines, dict)
    assert all(
        isinstance(p, ComponentPipeline) for p in pipeline.component_pipelines.values()
    )
    assert set(pipeline.component_pipelines.keys()) == set(
        component.name for component in pipeline.components
    )
