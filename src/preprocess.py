import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import re
import pandas as pd  # type: ignore
import numpy as np
import argparse
from ast import literal_eval

from sklearn.preprocessing import MultiLabelBinarizer  # type: ignore
import nltk  # type: ignore
from nltk.stem.wordnet import WordNetLemmatizer  # type: ignore
from nltk.corpus import stopwords  # type: ignore
import gensim.downloader  # type: ignore

nltk.download("stopwords")
nltk.download("wordnet")

from src.constants import (
    DATA_FOLDER,
    BASE_COLUMNS,
    GENRE_PERC_THRESHOLD,
    KEPT_STOPWORDS,
    KEPT_GENRES,
)

CSV_PATH = os.path.join(DATA_FOLDER, "raw", "movies_metadata.csv")
PKL_PATH = os.path.join(DATA_FOLDER, "movies.pkl")
STOPWORDS = [word for word in stopwords.words("english") if word not in KEPT_STOPWORDS]


def load_df(path: str = CSV_PATH) -> pd.DataFrame:
    """Load the input dataframe for training"""
    if not os.path.exists(path):
        raise FileNotFoundError(
            "Raw csv file is missing. Make sure you download and unzip it."
        )

    return pd.read_csv(
        path, usecols=BASE_COLUMNS, dtype={col: str for col in BASE_COLUMNS}
    )


def extract_genres(genre_str: str) -> list:
    """Extract the genres from the base data genre string"""
    return [genre_dict["name"] for genre_dict in literal_eval(genre_str)]


def identify_top_genres(
    input_genres: np.ndarray, threshold: float = GENRE_PERC_THRESHOLD
) -> list:
    """
    Given a list of all genre occurences, identify the top occuring ones past a certain threshold
    """
    genres, frequencies = np.unique(np.concatenate(input_genres), return_counts=True)
    freq_threshold = sum(frequencies) * threshold
    return [genre for genre, freq in zip(genres, frequencies) if freq > freq_threshold]


def text_clean(text: str, stop_words: list = STOPWORDS) -> str:
    """Strip any special characters, lemmatize and re-join the input string"""
    text = re.sub("[^a-zA-Z]+", " ", text)
    text = text.lower()
    lem = WordNetLemmatizer()
    text = [lem.lemmatize(word) for word in text.split(" ") if word not in stop_words]  # type: ignore
    return " ".join(text)


def sentence_to_embeddings(
    sentence: str, model: gensim.models.keyedvectors.KeyedVectors
) -> tuple:
    """
    Transform a sentence into an embedding. If there aren't any encodable words
    in the sentence, it returns an empty array.
    Args:
    - sentence: str
    - model: word2vec model
    Returns:
    - embedding: np.array of shape (3, 100)
    - missing: true if there weren't any encodable words in the sentence
    """
    embeddings = []
    for word in sentence.split(" "):
        if word:
            try:
                embeddings.append(model[word])
            except KeyError:
                pass
    if not embeddings:
        return np.array([]), True
    return (
        np.array(
            [
                np.mean(embeddings, axis=0),
                np.sum(embeddings, axis=0),
                np.max(embeddings, axis=0),
            ]
        ),
        False,
    )


def transform_single(
    title: str, overview: str, model: gensim.models.keyedvectors.KeyedVectors
) -> tuple:
    title_embeddings, title_missing = sentence_to_embeddings(text_clean(title), model)
    oview_embeddings, oview_missing = sentence_to_embeddings(text_clean(overview), model)
    return np.append(title_embeddings, oview_embeddings), (title_missing or oview_missing)


def target_encode(genres: list, target_genres: list) -> np.ndarray:
    return MultiLabelBinarizer(classes=target_genres).fit_transform(genres)


def target_decode(
    encoded_genres: np.ndarray, target_genres: list = KEPT_GENRES
) -> np.ndarray:
    return np.array(target_genres)[encoded_genres == 1.0]


def preprocess() -> None:

    df = load_df()
    df["genres"] = df.genres.apply(extract_genres)
    df.drop_duplicates(subset=["title", "overview"], inplace=True)
    df.dropna(inplace=True)

    # top_genres = identify_top_genres(df.genres.values)
    df["genres"] = df["genres"].apply(
        lambda x: np.intersect1d(x, KEPT_GENRES, assume_unique=True)
    )
    df = df[df.genres.apply(len) > 0]

    targets = target_encode(df.genres.values.tolist(), KEPT_GENRES)

    glove = gensim.downloader.load("glove-wiki-gigaword-100")

    X, y = [], []
    for title, overview, target in zip(df.title.values, df.overview.values, targets):
        embeddings, missing = transform_single(title, overview, model=glove)
        if missing:
            continue
        X.append(embeddings)
        y.append(target)

    # Dump the training data
    with open(os.path.join(DATA_FOLDER, "X.npy"), "wb") as f:
        np.save(f, np.array(X))

    with open(os.path.join(DATA_FOLDER, "y.npy"), "wb") as f:
        np.save(f, np.array(y))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preprocess", type=bool, default=False, help="Preprocess the raw dataset."
    )
    args = parser.parse_args()  # type: ignore
    if args.preprocess:  # type: ignore
        preprocess()
