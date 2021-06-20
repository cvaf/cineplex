from src.predict import predict
from src.config import Config
import pytest
from click.exceptions import BadParameter


@pytest.fixture
def dummy_cfg():
    return Config()


def test_predict_insufficient_error(dummy_cfg):
    title, overview = "arstarst", "tora na dume"
    with pytest.raises(BadParameter):
        predict(title, overview, dummy_cfg)


def test_predict(dummy_cfg):
    title = "no country for old men"
    overview = "Violence and mayhem ensue after a hunter stumbles upon a drug deal gone wrong and more than two million dollars in cash near the Rio Grande."
    genres_str = predict(title, overview, dummy_cfg)
    genres_list = genres_str.split("; ")
    assert len(genres_list) > 0
