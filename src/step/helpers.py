from pathlib import Path

from pandas.api.types import is_list_like


def _dict_to_list(d, key_list, default=None):

    l = []
    for key, value in d.items():
        if key in key_list:
            l.append(value)
        else:
            l.append(default)

    return l


def _get_participant_id(raw_file):
    """Generate a participant ID based on the raw file name(s)."""

    if is_list_like(raw_file):
        ids = [Path(elem).stem for elem in raw_file]
        participant_id = "_".join(ids)

    else:
        participant_id = Path(raw_file).stem

    return participant_id
