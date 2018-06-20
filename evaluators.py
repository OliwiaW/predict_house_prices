"""
this module contain all metrics we are going to use to evaluate our models
"""

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split


def count_r2(X, y, model):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return r2_score(y_test, y_pred)

def count_r2_and_mse_for_xgb(X_test, y_test, model):
    y_pred = model.predict(X_test)

    return r2_score(y_test, y_pred), mean_squared_error(y_test, y_pred)