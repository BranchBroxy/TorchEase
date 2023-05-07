from setuptools import setup, find_packages
import codecs
import os

DESCRIPTION = 'Simplifying working with PyTorch'
LONG_DESCRIPTION = 'TorchEase is a Python package that simplifies working with PyTorch, a popular deep learning framework. It provides a set of easy-to-use tools for training, testing, and deploying machine learning models. TorchEase includes features like early stopping, model fusing, and model evaluation, among others, which help streamline the model development process. Additionally, TorchEase is designed to be highly modular, making it easy to integrate into existing PyTorch projects.'

#with open(os.path.join(os.path.dirname(__file__), "TorchEase", "VERSION")) as version_file:
#    version = version_file.read().strip()
version = "0.0.13"

# Setting up
setup(
    name="TorchEase",
    version=version,
    author="Philipp Steigerwald",
    author_email="info@b-stream.info",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=["scikit-learn", "matplotlib", "pandas", "torch", "numpy", "scipy"],
    keywords=['python', 'torch', 'pytorch', 'early stop', 'trainer', 'tensor'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)