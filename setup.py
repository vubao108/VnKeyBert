from setuptools import setup, find_packages
base_packages = [
    "scikit-learn>=0.22.2",
    "numpy>=1.18.5",
    "rich>=10.4.0",
    "torch",
    "transformers",
    "py_vncorenlp",
    
]

use_packages = ["tensorflow", "tensorflow_hub", "tensorflow_text"]

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="vnkeybert",
    packages=find_packages(exclude=["notebooks", "docs"]),
    version="0.1.0",
    author="vubao",
    author_email="vubao108@gmail.com",
    description="VnKeyBERT performs Vietnamese keyword extraction with state-of-the-art transformer models.",
    long_description= long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vubao108/VnKeyBert",
    
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=base_packages,
    python_requires=">=3.6",
)
