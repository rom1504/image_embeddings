# image_embeddings
[![pypi](https://img.shields.io/pypi/v/image_embeddings.svg)](https://pypi.python.org/pypi/image_embeddings)
[![ci](https://github.com/rom1504/image_embeddings/workflows/Continuous%20integration/badge.svg)](https://github.com/rom1504/image_embeddings/actions?query=workflow%3A%22Continuous+integration%22)


Using efficientnet to provide embeddings for retrieval.

Why this repo ? Embeddings are a widely used technique that is well known in scientific circles. But it seems to be underused and not very well known for most engineers. I want to show how easy it is to represent things as embeddings, and how many application this unlocks.

## Workflow
1. download some pictures
2. run inference on them to get embeddings
3. simple knn example, to understand what's the point : click on some pictures and see KNN

## Simple Install

Run `pip install image_embeddings`

## Example workflow

1. run `image_embeddings save_examples_to_folder --output_folder=tf_flower_images`, this will retrieve the image files from https://www.tensorflow.org/datasets/catalog/tf_flowers (but you can also pick any other dataset)
2. run the inference
3. run the knn

## API

### image_embeddings.downloader.save_examples_to_folder(output_folder, dataset="tf_flowers")

Save https://www.tensorflow.org/datasets/catalog/tf_flowers to folder
Also works with other tf datasets

## Advanced Installation

### Prerequisites

Make sure you use `python>=3.6` and an up-to-date version of `pip` and
`setuptools`

    python --version
    pip install -U pip setuptools

It is recommended to install `image_embeddings` in a new virtual environment. For
example

    python3 -m venv image_embeddings_env
    source image_embeddings_env/bin/activate
    pip install -U pip setuptools
    pip install image_embeddings

### Using Pip

    pip install image_embeddings

### From Source

First, clone the `image_embeddings` repo on your local machine with

    git clone https://github.com/rom1504/image_embeddings.git
    cd image_embeddings
    make install

To install development tools and test requirements, run

    make install-dev

## Test

To run unit tests in your current environment, run

    make test

To run lint + unit tests in a fresh virtual environment,
run

    make venv-lint-test

## Lint

To run `black --check`:

    make lint

To auto-format the code using `black`

    make black

## Tasks

* [x] simple downloader in python
* [ ] simple inference in python using https://github.com/qubvel/efficientnet
* [ ] build python basic knn example using https://github.com/facebookresearch/faiss
* [ ] build basic ui using lit element and some brute force knn to show what it does, put in github pages
* [ ] use to illustrate embeddings blogpost
