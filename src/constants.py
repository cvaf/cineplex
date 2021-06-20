import os

# FOLDERS
PARENT_FOLDER = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)
DATA_FOLDER = os.path.join(PARENT_FOLDER, "data")
MODEL_FOLDER = os.path.join(PARENT_FOLDER, "models")

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
SENTENCE_EMBEDDING_SHAPE = (3, 100)
WORD_EMBEDDING_SHAPE = (100,)


# Modeling
INPUT_SIZE = 600
OUTPUT_SIZE = len(KEPT_GENRES)
LAYERS = [512, 256, 128]
EPOCHS = 10
PREDICTION_THRESHOLD = 0.5
