from setuptools import setup, find_packages

setup(
    name="simplegpt",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "transformers[torch]",
        "torchtune",
        "datasets",
        "pytest",
        "tiktoken",
        "wandb",
        "tqdm"
    ],
    python_requires='>=3.9',
)