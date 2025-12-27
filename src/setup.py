# setup.py
from setuptools import setup, find_packages

setup(
    name="wbc_classifier_app",
    version="0.1",
    py_modules=["model_loader", "predictor", "preprocessing", "reporting", "wbc_app", …],
    install_requires=[
        "streamlit",
        "tensorflow",
        "numpy",
        # etc.
    ],
)
