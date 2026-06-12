def test_group_pipeline(sample_group_pipeline):
    """Tests the GroupPipeline class."""

    assert len(sample_group_pipeline.participant_pipelines) == 2


def test_group_pipeline_besa(sample_group_pipeline_besa):
    """Tests the GroupPipeline class with BESA/MSEC correction."""

    assert len(sample_group_pipeline_besa.participant_pipelines) == 2
    assert sample_group_pipeline_besa.participant_pipelines[
        "09"
    ].preproc_pipeline.bad_channels == ["Fp1", "PO8"]
    assert (
        sample_group_pipeline_besa.participant_pipelines[
            "12"
        ].preproc_pipeline.bad_channels
        is None
    )
