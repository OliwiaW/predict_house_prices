"""
here we are going to run everything
"""

import prepare_data, models, visualisations

# prepare our data to train
X, y = prepare_data.prepare_data()

# let's see result for the easiest model - just simple linear regression
linear_regression_R2, linear_regression_MSE = models.count_linear_regression(X, y)
print('linear_regression_R2: ' , linear_regression_R2, ' linear_regression_MSE: ', linear_regression_MSE)

# let's check our regression with gradient boosting, first search for best parameters and that train our model on it
gbr_R2, gbr_MSE = models.train_best_gb_regression(X,y)
print('best_gb_regression_R2 : ', gbr_R2, ' best_gb_regression_MSE : ', gbr_MSE)

# let's check our regression with gradient boosting, first search for best parameters and that train our model on it
xgb_R2, xgb_MSE = models.train_best_extreme_gb(X, y)
print('best_XGBoost_R2 : ', xgb_R2, ' best_XGBoost_MSE : ', xgb_MSE)


# ok, now we can compare our models and visualize their scores
R2_scores = [linear_regression_R2, gbr_R2, xgb_R2]
MSE_scores = [abs(linear_regression_MSE), abs(gbr_MSE), abs(xgb_MSE)]
models_to_compare = ['linear_regression', 'Best GBRegressor', 'xgboost']

print('R2_scores: ', R2_scores)
print('MSE_scores: ', MSE_scores)

visualisations.plot_score(models_to_compare, R2_scores,'R2')
visualisations.plot_score(models_to_compare, MSE_scores,'MSE')

# we see that even our best score is not the best -> probably we should come back to the beginning
# and dive deeper into data, then if it won't help, we may play with different methods and it's parameters
# however it's not complicated case (just predicting price from numerical data) it might be a need to split
# the whole dataset on some specific clusters and create different models for each cluster
