# Cineplex - a movie genre classifier

## 1. How to use

**Clone**:

Clone the repository and navigate to the directory:
```bash
$ git clone git@github.com:cvaf/cineplex.git
$ cd cineplex
```

**Set-up**:

The dataset zip file has been kept in the repo to avoid having to set-up your kaggle credentials. Otherwise, you can run the following to download the zip file:
```bash
$ kaggle datasets download rounakbanik/the-movies-dataset -f movies_metadata.csv -p data/raw
```

Install poetry and the necessary dependencies and extract the data with:
```bash
$ make setup
```

**Using the classifier**:

```bash
$ poetry run python movie_classifier.py --title "Othello" --description "The evil Iago pretends to be friend of Othello in order to manipulate him to serve his own end in the film version of this Shakespeare classic."

 Title: Othello
 Description: The evil Iago pretends to be friend of Othello in order to manipulate him to serve his own end in the film version of this Shakespeare classic
 Genre(s): Comedy; Drama; Romance
```

## 2. Decisions

Considering we're dealing with text data, I used nltk to lemmatize and strip the dataset. In regards to the embeddings, I was between `tfidf` and `word2vec` and opted for the latter as I thought (perhaps mistakingly) that it would perform better.


Since most movies are part of multiple genres, the classifier has to be set-up accordingly -- ended up using a dense neural net with a binary cross entropy loss. To account for the class imbalance in the dataset (there's a lot of Drama movies), I added weights in the loss function. For tuning I tried different variations of layer sizes, including/removing batch processing as well as different activation functions. The resulting model ended up achieving an 81% [hamming score](https://en.wikipedia.org/wiki/Hamming_distance) on the test set, which honestly is way better than I expected.

For testing and type checking, I used python's default `pytest` and `mypy`. These were set-up as an automated action in github workflow. The test coverage isn't as high as I hoped (~70%), so that's the first thing I would improve if I had more time to work on it.

[Linting and other checks](.pre-commit-config.yaml) were set-up as a pre-commit hook to ensure code consistency.

## 3. Reproducing

Assuming `make setup` has been ran, the command below will preprocess the dataset and commence the training process with the default hyperparameters:

```bash
$ make train
```

You can modify the model parameters in the [config file](src/config.py).

Once the model is training, use the following to launch tensorboard for closer monitoring:
```bash
poetry run tensorboard --logdir ./results
```

The tests are set-up as an automated action on every push but you can also run them manually with:
```bash
$ make check
```
