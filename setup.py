from setuptools import setup, find_packages

setup(
    name="anomaly-detection",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=0.24.2",
        "torch>=1.9.0",
        "nltk>=3.6.2",
        "spacy>=3.1.0",
        "transformers>=4.5.0",
        "matplotlib>=3.4.2",
        "seaborn>=0.11.1"
    ]
) 