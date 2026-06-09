import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.train import trained_model
from sklearn.metrics import accuracy_score

def test_model_accuracy():
    model, x_test, y_test = trained_model()
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    assert accuracy >= 0.90

def test_prediction_shape():
    model, x_test, _ = trained_model()
    preds = model.predict(x_test)
    assert preds.shape == (len(x_test),)

def test_prediction_classes():
    model, x_test, _ = trained_model()
    preds = model.predict(x_test)
    assert set(preds).issubset({0, 1, 2})
