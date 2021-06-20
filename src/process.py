import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import re
import pandas as pd
import numpy as np
import pickle
import argparse
from ast import literal_eval

from sklearn.preprocessing import MultiLabelBinarizer
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import gensim.downloader

nltk.download("stopwords")
nltk.download("wordnet")

from src.constants import (
    DATA_FOLDER,
    MODEL_FOLDER,
    BASE_COLUMNS,
    GENRE_PERC_THRESHOLD,
    KEPT_STOPWORDS,
    KEPT_GENRES,
)

CSV_PATH = os.path.join(DATA_FOLDER, "raw", "movies_metadata.csv")
PKL_PATH = os.path.join(DATA_FOLDER, "movies.pkl")
STOPWORDS = [word for word in stopwords.words("english") if word not in KEPT_STOPWORDS]


def load_df() -> pd.DataFrame:
    """Load the input dataframe for training"""
    return pd.read_csv(
        CSV_PATH, usecols=BASE_COLUMNS, dtype={col: str for col in BASE_COLUMNS}
    )


def extract_genres(genre_str: str) -> list:
    """Extract the genres from the base data genre string"""
    return [genre_dict["name"] for genre_dict in literal_eval(genre_str)]


def identify_top_genres(
    input_genres: np.array, threshold: float = GENRE_PERC_THRESHOLD
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
    text = [lem.lemmatize(word) for word in text.split(" ") if word not in stop_words]
    return " ".join(text)


def create_embeddings(
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
    title_embeddings, title_missing = create_embeddings(text_clean(title), model)
    oview_embeddings, overview_missing = create_embeddings(text_clean(overview), model)
    return np.append(title_embeddings, oview_embeddings), (
        title_missing or overview_missing
    )


def fit_target_encoder(genres: list, target_genres: list) -> MultiLabelBinarizer:
    """Fit and transform the target, save the MLB locally"""
    mlb = MultiLabelBinarizer(classes=target_genres).fit(genres)
    with open(os.path.join(MODEL_FOLDER, "mlb.pkl"), "wb") as f:
        pickle.dump(mlb, f)
    return mlb


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preprocess", type=bool, default=False, help="Preprocess the raw dataset."
    )
    args = parser.parse_args()

    if not args.preprocess:
        sys.exit()

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            "Raw csv file is missing. Make sure you download and unzip it."
        )

    df = load_df()
    df["genres"] = df.genres.apply(extract_genres)
    df.drop_duplicates(subset=["title", "overview"], inplace=True)
    df.dropna(inplace=True)

    # top_genres = identify_top_genres(df.genres.values)
    df["genres"] = df["genres"].apply(
        lambda x: np.intersect1d(x, KEPT_GENRES, assume_unique=True)
    )

    mlb = fit_target_encoder(df.genres.values.tolist(), KEPT_GENRES)
    targets = mlb.transform(df.genres.values)

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
