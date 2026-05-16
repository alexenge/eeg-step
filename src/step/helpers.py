from pathlib import Path

from pandas.api.types import is_list_like

from .average import AverageConfig


def _process_files_input(files_input, file_extensions, n_out=None):
    """Process the `..._files` input arguments of the pipeline."""

    if is_list_like(files_input):
        return files_input

    elif isinstance(files_input, (str, Path)):
        files = []
        for ext in file_extensions:
            files.extend(Path(files_input).glob(f"*.{ext}"))
        assert len(files) > 0, (
            f"No files with extensions {file_extensions} found in folder {files_input}"
        )
        if n_out is not None:
            assert len(files) == n_out, (
                f"Number of files with extensions {file_extensions} in folder "
                f"{files_input} ({len(files)}) does not match the number of "
                f"participants ({n_out})"
            )
        return sorted(files)

    else:
        if n_out is not None:
            return [files_input] * n_out
        else:
            ValueError("If `files_input` is a scalar, `n_out` must be specified")


def _dict_to_list(input_dict, key_list, default=None):
    """Convert a dictionary to a list based on a list of keys."""

    output_list = []
    for key in key_list:
        if key in input_dict:
            output_list.append(input_dict[key])
        else:
            output_list.append(default)

    return output_list


def _get_participant_id(raw_file):
    """Generate a participant ID based on the raw file name(s)."""

    if is_list_like(raw_file):
        ids = [Path(elem).stem for elem in raw_file]
        participant_id = "_".join(ids)

    else:
        participant_id = Path(raw_file).stem

    return participant_id


def _dict_to_average_configs(input_dict):
    """Convert a dictionary to a list of AverageConfig objects.

    Dictionary keys are the names of the averages and dictionary values are the
    corresponding query strings."""

    average_configs = []
    for name, query in input_dict.items():
        average_config = AverageConfig(name=name, query=query)
        average_configs.append(average_config)

    return average_configs
