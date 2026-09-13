.PHONY: help install test collect digest offline check build stats clean

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "%-10s %s\n", $$1, $$2}'

install:  ## create .venv and install the package with dev extras
	python3 -m venv .venv
	.venv/bin/pip install -q --upgrade pip
	.venv/bin/pip install -q -e '.[dev]'

test:  ## run the test suite
	.venv/bin/pytest -q

collect:  ## daily pipeline: fetch, detect language, dedupe, store (no LLM)
	.venv/bin/news-digest collect

digest:  ## weekly pipeline: embed, cluster, LLM, rank, publish
	.venv/bin/news-digest digest

offline:  ## weekly pipeline with no API calls at all
	.venv/bin/news-digest digest --no-llm --no-embeddings

check:  ## fetch every enabled source once and report
	.venv/bin/news-digest sources --check

build:  ## regenerate docs/ from the database
	.venv/bin/news-digest build

stats:  ## summarize the database
	.venv/bin/news-digest stats

clean:  ## remove caches and local scratch databases
	rm -rf .pytest_cache **/__pycache__ data/local*.db
