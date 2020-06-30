from image_embeddings.downloader import save_examples_to_folder
import fire
import logging


def main():
    """Main entry point"""
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("tensorflow").handlers = []
    fire.Fire({"save_examples_to_folder": save_examples_to_folder})
