from image_embeddings.downloader import save_examples_to_folder
from image_embeddings.inference import write_tfrecord
from image_embeddings.inference import run_inference
from image_embeddings.knn import random_search, embeddings_to_numpy
import fire
import logging


def main():
    """Main entry point"""
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("tensorflow").handlers = []
    fire.Fire(
        {
            "save_examples_to_folder": save_examples_to_folder,
            "write_tfrecord": write_tfrecord,
            "run_inference": run_inference,
            "random_search": random_search,
            "embeddings_to_numpy": embeddings_to_numpy,
        }
    )
