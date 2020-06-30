install:
	python -m pip install -U pip setuptools wheel
	python -m pip install -r requirements.txt

install-dev: ## [Local development] Install test requirements
	python -m pip install -r requirements-test.txt

lint: ## [Local development] Run mypy, pylint and black
	python -m black --check -l 120 image_embeddings

test: ## [Local development] Run unit tests
	python -m pytest -v tests/unit

black: ## [Local development] Auto-format python code using black
	python -m black -l 120 .

build-dist: ## [Continuous integration] Build package for pypi
	python3.6 -m venv .env
	. .env/bin/activate && pip install -U pip setuptools wheel
	. .env/bin/activate && python setup.py sdist
	rm -rf .env

venv-lint-test: ## [Continuous integration] Install in venv and run lint and test
	python3.6 -m venv .env && . .env/bin/activate && make install install-dev lint test && rm -rf .env

.PHONY: help

help: # Run `make help` to get help on the make commands
	@grep -E '^[0-9a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'