import numpy as np
import pytest

from step.component import ComponentConfig, ComponentPipeline
from step.datasets.ucap import get_ucap
from step.epoch import EpochConfig, EpochPipeline
from step.input import InputConfig, InputPipeline
from step.participant import ParticipantConfig, ParticipantPipeline
from step.preproc import PreprocConfig, PreprocPipeline


@pytest.fixture(scope="session")
def sample_data():
    """Downloads some EEG data to use for running all tests."""

    return get_ucap(participants=["09", "12"])


@pytest.fixture(scope="session")
def sample_input_config(sample_data):
    """Creates an InputConfig for the sample data."""

    return InputConfig(
        raw_file=sample_data["raw_files"][0], log_file=sample_data["log_files"][0]
    )


@pytest.fixture(scope="session")
def sample_input_config_besa(sample_data):
    """Creates an InputConfig for the sample data incl. BESA file."""

    return InputConfig(
        raw_file=sample_data["raw_files"][0],
        log_file=sample_data["log_files"][0],
        besa_file=sample_data["besa_files"][0],
    )


@pytest.fixture(scope="session")
def sample_input_config_combine(sample_data):
    """Creates an InputConfig for the case when a participant has
    multiple EEG files that need to be combined."""

    return InputConfig(raw_file=sample_data["raw_files"][0:2])


@pytest.fixture(scope="session")
def sample_input_pipeline(sample_input_config):
    """Creates and runs an InputPipeline for the sample data."""

    input_pipeline = InputPipeline(sample_input_config)
    input_pipeline.run()

    return input_pipeline


@pytest.fixture(scope="session")
def sample_input_pipeline_besa(sample_input_config_besa):
    """Creates and runs an InputPipeline for the sample data using BESA."""

    input_pipeline = InputPipeline(sample_input_config_besa)
    input_pipeline.run()

    return input_pipeline


@pytest.fixture(scope="session")
def sample_input_pipeline_combine(sample_input_config_combine):
    """Creates and runs an InputPipeline for the case when a participant has
    multiple EEG files that need to be combined."""

    input_pipeline = InputPipeline(sample_input_config_combine)
    input_pipeline.run()

    return input_pipeline


@pytest.fixture(scope="session")
def sample_preproc_config():
    """Creates a PreprocConfig for the sample data using ICA and
    automatic bad channel detection."""

    return PreprocConfig(downsample_sfreq=100, bad_channels="auto")


@pytest.fixture(scope="session")
def sample_preproc_config_besa():
    """Creates a PreprocConfig for the sample data using BESA
    and manual bad channel selection."""

    return PreprocConfig(
        downsample_sfreq=100, bad_channels=["Fp1", "PO8"], ica_method=None
    )


@pytest.fixture(scope="session")
def sample_preproc_pipeline(sample_preproc_config, sample_input_pipeline):
    """Creates and runs a PreprocPipeline for the sample data using
    ICA."""

    preproc_pipeline = PreprocPipeline(sample_preproc_config)
    raw = sample_input_pipeline.raw
    preproc_pipeline.run(raw)

    return preproc_pipeline


@pytest.fixture(scope="session")
def sample_preproc_pipeline_besa(
    sample_preproc_config_besa, sample_input_pipeline_besa
):
    """Creates and runs a PreprocPipeline for the sample data using BESA
    and manual bad channel selection."""

    preproc_pipeline = PreprocPipeline(sample_preproc_config_besa)
    raw = sample_input_pipeline_besa.raw
    besa = sample_input_pipeline_besa.besa
    preproc_pipeline.run(raw, besa)

    return preproc_pipeline


@pytest.fixture(scope="session")
def sample_epoch_config():
    """Creates an EpochConfig for the sample data."""

    return EpochConfig(
        triggers=[
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
    )


@pytest.fixture(scope="session")
def sample_epoch_config_match():
    """Creates an EpochConfig for the sample data."""

    return EpochConfig(
        triggers=[
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
        ],
        triggers_column="bot",
    )


@pytest.fixture(scope="session")
def sample_epoch_pipeline(
    sample_epoch_config, sample_input_pipeline, sample_preproc_pipeline
):
    """Creates and runs an EpochPipeline for the sample data."""

    epoch_pipeline = EpochPipeline(sample_epoch_config)
    raw = sample_preproc_pipeline.raw
    log = sample_input_pipeline.log
    epoch_pipeline.run(raw, log)

    return epoch_pipeline


@pytest.fixture(scope="session")
def sample_epoch_pipeline_match(
    sample_epoch_config_match, sample_input_pipeline, sample_preproc_pipeline
):
    """Creates and runs an EpochPipeline for the sample data."""

    epoch_pipeline = EpochPipeline(sample_epoch_config_match)
    # Let's pretend some trials/triggers are missing from the EEG recording
    raw_to_match = sample_preproc_pipeline.raw.copy()
    ixs = np.concatenate(
        [
            np.arange(0, 1001),
            np.arange(5000, 6001),
            np.arange(10000, len(raw_to_match.annotations)),
        ]
    )
    raw_to_match.annotations.delete(ixs)
    log = sample_input_pipeline.log
    epoch_pipeline.run(raw_to_match, log)

    return epoch_pipeline


@pytest.fixture(scope="session")
def sample_component_config_n2():
    """Creates a ComponentConfig for the N2 component."""

    return ComponentConfig(
        name="N2",
        tmin=0.25,
        tmax=0.35,
        roi=["FC1", "FC2", "C1", "C2", "Cz"],
        compute_se=True,
    )


@pytest.fixture(scope="session")
def sample_component_config_p3b():
    """Creates a ComponentConfig for the P3b component."""

    return ComponentConfig(
        name="P3b",
        tmin=0.4,
        tmax=0.55,
        roi=["CP3", "CP1", "CPz", "CP2", "CP4", "P3", "Pz", "P4", "PO3", "POz", "PO4"],
        compute_se=False,
    )


@pytest.fixture(scope="session")
def sample_component_configs(sample_component_config_n2, sample_component_config_p3b):
    """Creates a list of ComponentConfigs for the sample data."""

    return [sample_component_config_n2, sample_component_config_p3b]


@pytest.fixture(scope="session")
def sample_component_pipeline(sample_component_config_n2, sample_epoch_pipeline):
    """Creates and runs a ComponentPipeline for the sample data."""

    component_pipeline = ComponentPipeline(sample_component_config_n2)

    epochs = sample_epoch_pipeline.epochs
    bad_ixs = sample_epoch_pipeline.bad_ixs

    component_pipeline.run(epochs, bad_ixs)

    return component_pipeline


@pytest.fixture(scope="session")
def sample_participant_config(
    sample_input_config,
    sample_preproc_config,
    sample_epoch_config,
    sample_component_configs,
):
    """Creates a ParticipantConfig for the sample data."""

    return ParticipantConfig(
        input_config=sample_input_config,
        preproc_config=sample_preproc_config,
        epoch_config=sample_epoch_config,
        component_configs=sample_component_configs,
    )


@pytest.fixture(scope="session")
def sample_participant_pipeline(sample_participant_config):
    """Creates and runs a ParticipantPipeline for the sample data."""

    participant_pipeline = ParticipantPipeline(sample_participant_config)
    participant_pipeline.run()

    return participant_pipeline
