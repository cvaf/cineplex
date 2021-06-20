import pytest
from src.model import load_data
from tempfile import TemporaryDirectory


def test_load_data_directory():
    with TemporaryDirectory() as tempdir:
        with pytest.raises(FileNotFoundError):
            load_data(tempdir)
