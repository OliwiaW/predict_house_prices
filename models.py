"""
module for training our models
"""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import xgboost as xgb

from settings import NUMBER_OF_CV_FOLDS, SCORINGS, PARAMETERS, XGBOOST_PARAMETERS
from evaluators import count_r2, count_r2_and_mse_for_xgb


def count_linear_regression(X, y):
    lr = LinearRegression()
    # let's count average coefficient of determination (r2) so that we know how much our model can explain
    # and mean_squared_error so that we know how good is our model -> we will compare it with other models
    scores = cross_validate(estimator=lr, cv=NUMBER_OF_CV_FOLDS, X=X, y=y, scoring=SCORINGS)

    return count_r2(X, y, lr), round(np.average(scores['test_score']))


def train_best_gb_regression(X, y):
    # gb regressor will use various decision trees to make our prediction better
    gb = GradientBoostingRegressor()
    # GridSearchCV will choose for us best parameters from our PARAMETERS
    gs_cv_gb = GridSearchCV(estimator=gb, param_grid=PARAMETERS, scoring=SCORINGS)
    gs_cv_gb = gs_cv_gb.fit(X, y)
    # let's see our best parameters and send it to the next function to train the best model one more time using cv
    print('best parameters for Gradient Boosting Regressor:')
    for k, v in zip(gs_cv_gb.best_params_.keys(), gs_cv_gb.best_params_.values()):
        print(k, " : ", v)
    print('best MSE: ', gs_cv_gb.best_score_)

    # fyi best parameters:
    # {'learning_rate': 0.05, 'loss': 'ls', 'max_depth': 5, 'min_samples_leaf': 10, 'n_estimators': 300}

    gb = GradientBoostingRegressor(learning_rate=gs_cv_gb.best_params_['learning_rate'],
                                   max_depth=gs_cv_gb.best_params_['max_depth'],
                                   min_samples_leaf=gs_cv_gb.best_params_['min_samples_leaf'],
                                   n_estimators=gs_cv_gb.best_params_['n_estimators'])
    scores = cross_validate(estimator=gb, cv=NUMBER_OF_CV_FOLDS, X=X, y=y, scoring=SCORINGS)

    return count_r2(X, y, gb), round(np.average(scores['test_score']))


def train_best_extreme_gb(X, y):
    # we can also use another library for xgboost, let's check it's results
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    Xy_train = xgb.DMatrix(X_train.values, y_train, feature_names=X_train.columns.values)
    X_test = xgb.DMatrix(X_test.values, feature_names=X_test.columns.values)

    # we should do cross validation here or GridSearchCV as in functions above, but let's do it quickly for now
    model = xgb.train(dtrain=Xy_train, params=XGBOOST_PARAMETERS)
    r2, mse = count_r2_and_mse_for_xgb(X_test, y_test, model)

    return r2, mse
