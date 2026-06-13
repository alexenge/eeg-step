from pathlib import Path

from step.group import GroupPipeline
from step.participant import ParticipantPipeline


def test_group_pipeline(sample_group_pipeline):
    """Tests the GroupPipeline class."""

    pipeline = sample_group_pipeline
    assert isinstance(pipeline, GroupPipeline)

    assert isinstance(pipeline.raw_files_, list)
    assert all(isinstance(raw_file, (str, Path)) for raw_file in pipeline.raw_files_)

    assert isinstance(pipeline.participant_ids, list)

    assert isinstance(pipeline.log_files_, list)
    assert all(isinstance(log_file, (str, Path)) for log_file in pipeline.log_files_)

    assert isinstance(pipeline.besa_files_, list)
    assert all(besa_file is None for besa_file in pipeline.besa_files_)

    assert isinstance(pipeline.bad_channels_, list)
    assert all(
        bad_channels is None or isinstance(bad_channels, list)
        for bad_channels in pipeline.bad_channels_
    )

    participant_pipelines = pipeline.participant_pipelines
    assert isinstance(participant_pipelines, dict)
    assert len(participant_pipelines) == 2
    for participant_id, participant_pipeline in participant_pipelines.items():
        assert participant_id in pipeline.participant_ids
        assert isinstance(participant_pipeline, ParticipantPipeline)


def test_group_pipeline_besa(sample_group_pipeline_besa):
    """Tests the GroupPipeline class with BESA/MSEC correction."""

    pipeline = sample_group_pipeline_besa
    assert isinstance(pipeline, GroupPipeline)

    assert isinstance(pipeline.besa_files_, list)
    assert all(isinstance(besa_file, (str, Path)) for besa_file in pipeline.besa_files_)

    participant_pipelines = pipeline.participant_pipelines
    assert len(participant_pipelines) == 2

    assert participant_pipelines["09"].preproc_pipeline.bad_channels == ["Fp1", "PO8"]
    assert participant_pipelines["12"].preproc_pipeline.bad_channels is None
