import os

PARENT_FOLDER = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)
DATA_FOLDER = os.path.join(PARENT_FOLDER, "data")

BASE_COLUMNS = ["title", "overview", "genres"]


GENRE_PERC_THRESHOLD = 0.03

TFIDF_MAX_FEATURES = [100, 1000]
