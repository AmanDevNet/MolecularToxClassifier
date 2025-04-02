MolecularToxClassifier

MolecularToxClassifier is a machine learning-based project that predicts the toxicity of molecular compounds using the DeepChem library. It utilizes molecular fingerprints and regression models to estimate toxicity levels.

Features:

Uses DeepChem for molecular data processing.

Implements Morgan Fingerprint as a molecular featurizer.

Trains a regression model to predict toxicity.

Computes Mean Squared Error (MSE) for model evaluation.

Installation:

Ensure you have Python 3.8+ and install the required dependencies:

pip install deepchem scikit-learn numpy pandas rdkit tensorflow

Usage:

Run the Python script to train and evaluate the model:

python MolecularToxClassifier.py

Project Structure:

MolecularToxClassifier/
│── MolecularToxClassifier.py  # Main script
│── data/                      # Dataset files
│── models/                    # Trained models
│── README.md                  # Project documentation

Output:

The script outputs:

Model training progress

Mean Squared Error (MSE) on the dataset

Dependencies:

DeepChem for molecular feature extraction

RDKit for chemical structure handling

Scikit-learn for machine learning models

TensorFlow for deep learning support (if needed)

