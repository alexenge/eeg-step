from pathlib import Path

from pandas.api.types import is_list_like

from .average import AverageConfig


def _process_files_input(files_input, file_extensions):
    """Process the "files" input arguments of the pipeline from a list or folder
    path."""

    if is_list_like(files_input):
        return files_input
    else:
        files = []
        for ext in file_extensions:
            files.extend(Path(files_input).glob(f"*.{ext}"))
        return files


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
