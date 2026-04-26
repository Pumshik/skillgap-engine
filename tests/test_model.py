import numpy as np
from src.models.classical import MyLogisticRegression

def test_model_api():
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)
    model = MyLogisticRegression(max_iter=50)
    model.fit(X, y)
    
    assert hasattr(model, "coef_") and hasattr(model, "intercept_")
    assert model.predict(X).shape == (100,)
    assert model.predict_proba(X).shape == (100, 2)
    assert len(model.loss_history_) > 0