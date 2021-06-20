# Cineplex - a movie genre classifier

## How to use

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
 Genre(s): Adventure; Romance  
```

## Decisions

Considering we're dealing with text data, I used nltk to lemmatize and strip the dataset. In regards to the embeddings, I was between `tfidf` and `word2vec` and opted for the latter as I thought (perhaps mistakingly) that it would perform better.

For modeling I used a simple neural net with batch processing -- I realized it wouldn't do great performance-wise as it fails to capture context in sequential data, but it was the easiest to set-up for the purpose of our project. No tuning was done on the model, but if that had been my focus I would have set-up tensorboard to allow for closer monitoring of the training process.

For testing and type checking, I used python's default `pytest` and `mypy`. These were set-up as an automated action in github workflow. The test coverage isn't as high as I hoped, that's the first thing I would improve if I had more time to work on it.

[Linting and other checks](.pre-commit-config.yaml) were set-up as a pre-commit hook to ensure code consistency.

## Reproducing

Assuming `make setup` has been ran and the raw dataset has been extracted, running the command below will preprocess the dataset and commence the training process:
```bash
$ make train
```
You can modify the model parameters in the [config file](src/config.py).


The tests are set-up as an automated action on every push but you can also run them manually with:
```bash
$ make check
```
