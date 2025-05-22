
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def train_model(X, y):
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)
    return model

def classify_image(model, image):
    height, width = image.shape[1], image.shape[2]
    features = image.reshape(image.shape[0], -1).T
    prediction = model.predict(features)
    return prediction.reshape(height, width)
