def test_group_pipeline(sample_group_pipeline):
    """Tests the GroupPipeline class."""

    assert True


def test_group_pipeline_bad_channels(sample_group_pipeline_bad_channels):
    """Tests the GroupPipeline class with bad channels for one participant."""

    assert len(sample_group_pipeline_bad_channels.participant_pipelines) == 2
    assert sample_group_pipeline_bad_channels.participant_pipelines[
        "09"
    ].preproc_pipeline.config.bad_channels == ["Fp1", "PO8"]
    assert (
        sample_group_pipeline_bad_channels.participant_pipelines[
            "12"
        ].preproc_pipeline.config.bad_channels
        == []
    )
