# image-embeddings
Using efficientnet to provide embeddings for retrieval.

Why this repo ? Embeddings are a widely used technique that is well known in scientific circles. But it seems to be underused and not very well known for most engineers. I want to show how easy it is to represent things as embeddings, and how many application this unlocks.

## Workflow
1. download some pictures
2. run inference on them to get embeddings
3. simple knn example, to understand what's the point : click on some pictures and see KNN

## Tasks

* [ ] simple downloader in python
* [ ] simple inference in python using https://github.com/qubvel/efficientnet
* [ ] build python basic knn example using https://github.com/facebookresearch/faiss
* [ ] build basic ui using lit element and some brute force knn to show what it does, put in github pages
* [ ] use to illustrate embeddings blogpost
