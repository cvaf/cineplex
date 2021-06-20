setup:
	pip install poetry
	poetry install
	poetry run python -c 'import nltk; nltk.download("stopwords"); nltk.download("wordnet")'
	poetry run python -c 'import gensim.downloader; glove = gensim.downloader.load("glove-wiki-gigaword-100")'
	unzip data/raw/movies_metadata.csv.zip -d data/raw
	poetry run pre-commit install

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test:
	rm -f .coverage
	rm -f coverage.*

clean: clean-pyc clean-test


test: clean
	poetry run pytest --cov-config=.coveragerc --cov=src --cov-report xml

mypy:
	poetry run mypy src

check: test mypy


train:
	poetry run python movie_classifier.py --train
