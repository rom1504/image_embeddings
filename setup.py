"""Setup script."""

from pathlib import Path
import re
import setuptools


if __name__ == "__main__":
    # Read metadata from version.py
    with Path("image-embeddings/version.py").open(encoding="utf-8") as file:
        metadata = dict(re.findall(r'__([a-z]+)__\s*=\s*"([^"]+)"', file.read()))

    # Read description from README
    with Path(Path(__file__).parent, "docs", "README.rst").open(encoding="utf-8") as file:
        long_description = file.read()

    # Run setup
    setuptools.setup(
        name="image-embeddings",
        author=metadata["author"],
        version=metadata["version"],
        install_requires=["fire>=0.3", "numpy", "pandas>=1", "pyarrow>=0.14", "tensorflow>=2.2"],
        tests_require=["pytest"],
        dependency_links=[],
        data_files=[(".", ["requirements.txt", "README.md"])],
        packages=setuptools.find_packages(),
        description=long_description.split("\n")[0],
        long_description=long_description,
        long_description_content_type="text/x-markdown",
        classifiers=[
            "License :: OSI Approved :: MIT",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.6",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "Development Status :: 5 - Production/Stable",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
    )
