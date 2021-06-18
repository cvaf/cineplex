import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import pandas as pd
import numpy as np
from ast import literal_eval

from src.constants import DATA_FOLDER, BASE_COLUMNS, GENRE_PERC_THRESHOLD

CSV_PATH = os.path.join(DATA_FOLDER, "raw", "movies_metadata.csv")
PKL_PATH = os.path.join(DATA_FOLDER, "movies.pkl")


def load_df() -> pd.DataFrame:
    """Load the input dataframe for training"""
    return pd.read_csv(CSV_PATH, usecols=BASE_COLUMNS)


def extract_genres(genre_str: str) -> list:
    """Extract the genres from the base data genre string"""
    return [genre_dict["name"] for genre_dict in literal_eval(genre_str)]


def identify_top_genres(
    input_genres: np.array, threshold: float = GENRE_PERC_THRESHOLD
) -> list:
    genres, frequencies = np.unique(np.concatenate(input_genres), return_counts=True)
    freq_threshold = sum(frequencies) * threshold
    return [genre for genre, freq in zip(genres, frequencies) if freq > freq_threshold]


def genre_selection(genres: list, top_genres: list) -> list:
    return [genre for genre in genres if genre in top_genres]


def preprocess_training(data: pd.DataFrame) -> pd.DataFrame:
    """Data selection and processing methods for base data"""

    # Extract the genre information
    data["genres"] = data.genres.apply(extract_genres)

    data.drop_duplicates(subset=["title", "overview"], inplace=True)
    data.dropna(inplace=True)

    # Identify and remove observations with no top genres
    top_genres = identify_top_genres(data.genres.values)
    data["genres"] = data["genres"].apply(lambda x: genre_selection(x, top_genres))
    data = data[data.genres.apply(len) > 0]



    return


if __name__ == "__main__":

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            "Raw csv file is missing. Make sure you download and unzip it."
        )

    df = load_df()
    df["genres"] = df.genres.apply(extract_genres)

    top_genres = identify_top_genres(df.genres.values)
    # df["genres"] = 

    df.to_pickle(PKL_PATH)
