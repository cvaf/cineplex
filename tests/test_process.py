from src.process import (
    create_embeddings,
    extract_genres,
    load_df,
    identify_top_genres,
    text_clean,
)
from src.constants import SENTENCE_EMBEDDING_SHAPE, WORD_EMBEDDING_SHAPE
import numpy as np
import pytest


@pytest.fixture
def dummy_glove():
    return {
        "there": np.random.rand(*WORD_EMBEDDING_SHAPE),
        "hello": np.random.rand(*WORD_EMBEDDING_SHAPE),
    }


def test_extract_genres() -> None:
    genre_inputs = [
        "[{'id': 16, 'name': 'Animation'}, {'id': 35, 'name': 'Comedy'}, {'id': 10751, 'name': 'Family'}]",
        "[{'id': 12, 'name': 'Adventure'}, {'id': 14, 'name': 'Fantasy'}, {'id': 10751, 'name': 'Family'}]",
        "[]",
    ]

    genre_outputs = [
        ["Animation", "Comedy", "Family"],
        ["Adventure", "Fantasy", "Family"],
        [],
    ]
    for genre_input, genre_output in zip(genre_inputs, genre_outputs):
        assert extract_genres(genre_input) == genre_output


def test_load_df() -> None:
    df = load_df()
    assert df.shape == (45466, 3), "Input dataframe has incorrect shape."


def test_identify_top_genres() -> None:
    input_genres = np.array(
        [["Action"] * 70] + [["Comedy"] * 28] + [["Fantasy"] * 2], dtype=object
    )
    data = [
        (0.03, ["Action", "Comedy"]),
        (0.01, ["Action", "Comedy", "Fantasy"]),
        (0.5, ["Action"]),
        (0.75, []),
    ]
    for (threshold, output) in data:
        assert output == identify_top_genres(input_genres, threshold)


def test_text_clean():
    data = [
        ("Led by Woody, Andy's toys live happily", "led woody andy toy live happily"),
        ("It's the year 3000 AD. The world's most", "year ad world"),
        ("Just when George Banks has recovered", "george bank recovered"),
    ]

    for input, output in data:
        assert text_clean(input) == output


def test_create_embeddings(dummy_glove):
    data = [
        ("arstarst fhiaoien", (0,)),
        ("hello there", SENTENCE_EMBEDDING_SHAPE),
        ("arstarst hello", SENTENCE_EMBEDDING_SHAPE),
    ]

    for input, output_shape in data:
        output, _ = create_embeddings(input, dummy_glove)
        print(input, output.shape)
        assert output.shape == output_shape

    sentence_a = "hello arstarst"
    sentence_b = "arstarst hello"
    embedding_a, _ = create_embeddings(sentence_a, dummy_glove)
    embedding_b, _ = create_embeddings(sentence_b, dummy_glove)
    np.testing.assert_array_equal(embedding_a, embedding_b)
