import numpy as np
import pandas as pd
from mne.channels import DigMontage
from mne.io import BaseRaw

from step.preproc import AUTO_HEOG_CHANNELS, AUTO_VEOG_CHANNELS, PreprocPipeline


def test_preproc_pipeline(sample_preproc_pipeline, sample_raw):
    """Tests the PreprocPipeline class with ICA correction."""

    pipeline = sample_preproc_pipeline

    assert isinstance(pipeline, PreprocPipeline)

    assert pipeline.downsample_sfreq == 100
    assert pipeline.heog_channels == "auto"
    assert pipeline.veog_channels == "auto"
    assert pipeline.montage == "easycap-M1"
    # Some bad channels were auto-detected
    assert len(pipeline.bad_channels) > 0
    assert pipeline.ref_channels == "average"
    assert pipeline.ica_method == "fastica"
    assert pipeline.ica_n_components is None
    assert pipeline.ica_eog_channels == "auto"
    assert pipeline.highpass_freq == 0.1
    assert pipeline.lowpass_freq == 40.0

    assert isinstance(pipeline.raw, BaseRaw)

    assert pipeline.ica_method_ == "fastica"

    assert pipeline.raw.info["sfreq"] == 100.0

    assert pipeline.heog_channels_ == AUTO_HEOG_CHANNELS
    assert "HEOG" in pipeline.raw.ch_names
    assert pipeline.raw.get_channel_types(["HEOG"])[0] == "eog"

    assert pipeline.veog_channels_ == AUTO_VEOG_CHANNELS
    assert "VEOG" in pipeline.raw.ch_names
    assert pipeline.raw.get_channel_types(["VEOG"])[0] == "eog"

    assert isinstance(pipeline.montage_, DigMontage)
    # All EEG channels should have locations set via montage
    assert all(
        ~np.isnan(ch["loc"]).all() for ch in pipeline.raw.info["chs"] if ch["kind"] == 2
    )

    # No bad channels remain after interpolation
    assert pipeline.raw.info["bads"] == []

    assert pipeline.raw.info["custom_ref_applied"]

    assert pipeline.ica_eog_channels_ == ["HEOG", "VEOG"]
    assert len(pipeline.ica.labels_["eog/0/HEOG"]) > 0
    assert len(pipeline.ica.labels_["eog/1/VEOG"]) > 0

    assert pipeline.raw.info["highpass"] == 0.1
    assert pipeline.raw.info["lowpass"] == 40.0

    # Preprocessing should reduce the average standard deviation of the data
    std_input = sample_raw.get_data().std(axis=1).mean()
    std_preprocessed = pipeline.raw.get_data().std(axis=1).mean()
    assert std_input / std_preprocessed > 2.0


def test_preproc_pipeline_besa(sample_preproc_pipeline_besa, sample_raw_besa):
    """Tests the PreprocPipeline class with BESA correction."""

    pipeline = sample_preproc_pipeline_besa

    assert isinstance(pipeline, PreprocPipeline)

    assert isinstance(pipeline.raw, BaseRaw)
    assert pipeline.ica_method_ is None
    assert isinstance(pipeline.besa, pd.DataFrame)

    # Preprocessing should reduce the average standard deviation of the data
    std_input = sample_raw_besa.get_data().std(axis=1).mean()
    std_preprocessed = pipeline.raw.get_data().std(axis=1).mean()
    assert std_input / std_preprocessed > 1.5
