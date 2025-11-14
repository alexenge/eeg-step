import os

from step.datasets import get_erpcore, get_ucap


def test_get_erpcore():
    """Test the function for getting the ERP CORE dataset."""

    erpcore_files = get_erpcore(component="N400", participants=1)

    assert "raw_files" in erpcore_files
    assert "log_files" in erpcore_files
    assert len(erpcore_files["raw_files"]) == 1
    assert len(erpcore_files["log_files"]) == 1
    assert all(isinstance(f, str) for f in erpcore_files["raw_files"])
    assert all(isinstance(f, str) for f in erpcore_files["log_files"])
    assert all(os.path.isfile(f) for f in erpcore_files["raw_files"])
    assert all(os.path.isfile(f) for f in erpcore_files["log_files"])


def test_get_ucap():
    """Test the function for getting the UCAP dataset."""

    ucap_files = get_ucap(participants=["09", "12"])

    assert "raw_files" in ucap_files
    assert "log_files" in ucap_files
    assert "besa_files" in ucap_files
    assert len(ucap_files["raw_files"]) == 2
    assert len(ucap_files["log_files"]) == 2
    assert len(ucap_files["besa_files"]) == 2
    assert all(isinstance(f, str) for f in ucap_files["raw_files"])
    assert all(isinstance(f, str) for f in ucap_files["log_files"])
    assert all(isinstance(f, str) for f in ucap_files["besa_files"])
    assert all(os.path.isfile(f) for f in ucap_files["raw_files"])
    assert all(os.path.isfile(f) for f in ucap_files["log_files"])
    assert all(os.path.isfile(f) for f in ucap_files["besa_files"])
