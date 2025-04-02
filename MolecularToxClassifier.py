import deepchem as dc
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Corrected featurizer (Morgan Fingerprint)
featurizer = dc.feat.CircularFingerprint()  # ✅ Corrected Fix
tasks, datasets, transformers = dc.molnet.load_delaney(featurizer=featurizer)

train_dataset, valid_dataset, test_dataset = datasets

# Convert dataset to NumPy arrays
X_train, y_train = train_dataset.X, train_dataset.y.ravel()
X_valid, y_valid = valid_dataset.X, valid_dataset.y.ravel()
X_test, y_test = test_dataset.X, test_dataset.y.ravel()

# Train a Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")
