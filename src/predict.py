import gensim.downloader  # type: ignore
from click.exceptions import BadParameter
from gensim.models.keyedvectors import KeyedVectors  # type: ignore

from .config import Config
from .model import Trainer
from .preprocess import transform_single, target_decode


def predict(
    title: str,
    description: str,
    config: Config,
    glove: KeyedVectors = gensim.downloader.load("glove-wiki-gigaword-100"),
) -> str:
    trainer = Trainer(*config.model_params())

    X, missing_embeddings = transform_single(title, description, glove)

    if missing_embeddings:
        raise BadParameter("Title and overview have insufficient information")

    encoded_prediction = trainer.predict(X)
    prediction = target_decode(encoded_prediction)

    return "; ".join(prediction)
