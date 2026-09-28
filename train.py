# from sklearn.datasets import load_iris
# from sklearn.ensemble import RandomForestClassifier
# import joblib

# X, y = load_iris(return_X_y=True)
# model = RandomForestClassifier(n_estimators=100)
# model.fit(X, y)
# joblib.dump(model, "model/model.joblib")

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
X, y = load_iris(return_X_y=True)

# Ambil 2 fitur pertama saja (sepal_length & sepal_width)
X = X[:, :2]

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

# Simpan model baru
joblib.dump(model, "model/model.joblib")
print("Model berhasil dilatih ulang dengan 2 fitur!")