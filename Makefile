.PHONY: help install test run fetch parse digest offline check inspect build stats clean

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "%-10s %s\n", $$1, $$2}'

install:  ## create .venv and install the package with dev extras
	python3 -m venv .venv
	.venv/bin/pip install -q --upgrade pip
	.venv/bin/pip install -q -e '.[dev]'

test:  ## run the test suite
	.venv/bin/pytest -q

run:  ## the whole thing: fetch, parse, cluster, summarize, publish
	.venv/bin/news-digest run

fetch:  ## read the mailbox into the database
	.venv/bin/news-digest fetch

parse:  ## extract articles from stored messages
	.venv/bin/news-digest parse

digest:  ## cluster, summarize, rank and write, from what is already stored
	.venv/bin/news-digest digest

offline:  ## the weekly half with no API calls at all
	.venv/bin/news-digest digest --no-llm --no-embeddings

check:  ## do the source match rules actually match your mail?
	.venv/bin/news-digest sources --check

inspect:  ## what is stored for the last finished week; writes nothing
	.venv/bin/news-digest inspect

build:  ## regenerate digests/ from the database
	.venv/bin/news-digest build

stats:  ## summarize the database
	.venv/bin/news-digest stats

clean:  ## remove caches, debug output and scratch databases
	rm -rf .pytest_cache **/__pycache__ debug data/local*.db
