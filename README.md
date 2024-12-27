<div align="center">

# PhenoSeeker

[![python](https://img.shields.io/badge/-Python_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0.1-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
[![isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

</div>

## Description
PhenoSeeker - A Python toolkit for phenotype-based molecule discovery using Cell Painting data.

## Installation

1. **Clone the Repository**  
   Clone the PhenoSeeker repository to your local machine:
   ```bash
   git clone https://github.com/mxfly14/phenoseeker.git
   cd phenoseeker

2. **Set Up a Virtual Environment**  
   Create and activate a Python 3.10 virtual environment:
   ```bash
   python3.10 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

3. **Install Dependencies**  
   Install all required dependencies using Poetry:
   ```bash
   poetry install
   poetry shell

4. **Activate Poetry Shell**  
   Run poetry shell to enter the virtual environment:
   ```bash
   poetry shell

## Extracting Image Features

To extract image features using PhenoSeeker, follow these steps:

1. **Prepare the Configuration File**  
   Update the `configs/config_extraction.yaml` file with the appropriate paths and parameters for your dataset and feature extraction settings.

2. **Run the Extraction Script**  
   Execute the `extract_features.py` script located in the `scripts` directory:

   ```bash
   python scripts/extract_features.py

## Creating compounds phenotypic profiles

## Evaluating phenotypic profiles for molecule selection