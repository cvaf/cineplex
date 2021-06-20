import pytest
import numpy as np
from click.exceptions import BadParameter

from src.predict import predict
from src.config import Config
from src.constants import WORD_EMBEDDING_SHAPE


@pytest.fixture
def dummy_cfg():
    return Config()


@pytest.fixture
def dummy_glove():
    return {
        "there": np.random.rand(*WORD_EMBEDDING_SHAPE).astype("f"),
        "hello": np.random.rand(*WORD_EMBEDDING_SHAPE).astype("f"),
    }


def test_predict_insufficient_error(dummy_cfg, dummy_glove):
    title, overview = "arstarst", "tora na dume"
    with pytest.raises(BadParameter):
        predict(title, overview, dummy_cfg, dummy_glove)


# def test_predict(dummy_cfg, dummy_glove):
#     title = "hello there"
#     overview = "hello there hello there hello there"
#     genres_str = predict(title, overview, dummy_cfg, dummy_glove)
#     genres_list = genres_str.split("; ")
#     assert len(genres_list) > 0
