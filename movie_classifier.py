import os
from src.constants import MODEL_FOLDER
import click
import src


@click.command()
@click.option(
    "--preprocess", is_flag=True, default=False, help="run pre-processing script"
)
@click.option("--train", is_flag=True, default=False, help="train the model")
@click.option("--preload", is_flag=True, default=False, help="preload a trained model")
@click.option("--title", type=str, default=None, help="movie title to predict")
@click.option("--description", type=str, default=None, help="movie overview to predict")
def run(
    preprocess: bool, train: bool, preload: bool, title: str, description: str
) -> None:

    if (title and not description) or (not title and description):
        raise click.exceptions.BadParameter(
            "You must provide both the movie title and description for genre classification"
        )

    config = src.Config(preload=preload)

    if preprocess:
        src.preprocess()

    if train:
        if not all([f"{file}.npy" in os.listdir(src.DATA_FOLDER) for file in ["X", "y"]]):
            src.preprocess()
        src.train(config)

    if title and description:
        if "model.pt" not in os.listdir(MODEL_FOLDER):
            raise click.exceptions.BadParameter(
                "You must train a model before using the classifier."
            )
        genres = src.predict(title, description, config)
        print(f" Title: {title}\n Description: {description}\n Genre(s): {genres}")


if __name__ == "__main__":
    run()
