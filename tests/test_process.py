from src.process import extract_genres, load_df, identify_top_genres
import numpy as np


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
    input_genres = np.array([["Action"]*70] + [["Comedy"]*28] + [["Fantasy"]*2], dtype=object)
    
    # THRESHOLD, OUTPUT
    outputs = [
        (0.03, ["Action", "Comedy"]),
        (0.01, ["Action", "Comedy", "Fantasy"]),
        (0.5, ["Action"]),
        (0.75, []),
    ]
    for (threshold, output) in outputs:
        assert output == identify_top_genres(input_genres, threshold)

    