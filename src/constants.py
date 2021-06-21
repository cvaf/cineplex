import os

# FOLDERS
PARENT_FOLDER = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)
DATA_FOLDER = os.path.join(PARENT_FOLDER, "data")
MODEL_FOLDER = os.path.join(PARENT_FOLDER, "models")
RESULTS_FOLDER = os.path.join(PARENT_FOLDER, "results")

BASE_COLUMNS = ["title", "overview", "genres"]


GENRE_PERC_THRESHOLD = 0.03

TFIDF_MAX_FEATURES = [100, 1000]
KEPT_GENRES = [
    "Action",
    "Adventure",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Family",
    "Horror",
    "Romance",
    "Science Fiction",
    "Thriller",
]
KEPT_STOPWORDS = [
    "now",
    "then",
    "before",
    "after",
    "only",
    "you",
    "not",
    "all",
    "me",
    "be",
    "there",
]
WORD_EMBEDDING_SHAPE = (100,)
