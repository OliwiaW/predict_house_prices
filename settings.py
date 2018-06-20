"""
if files will have similar value distribution, there should not be a need to change values (so might be static),
in other case here is a place, where user can set various parameters
"""

# for module 'prepare_data':
FILE_PATH = 'house.csv'
MAX_PRICE_THERESHOLD = 3000000
COLUMNS_TO_KEEP = ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', \
                   'view', 'grade', 'sqft_above', 'sqft_basement', 'is_renovated']

# for module 'models':
NUMBER_OF_CV_FOLDS = 6
SCORINGS =  'neg_mean_squared_error'

# just set some values
PARAMETERS ={'loss': ['ls'], # least-squered, ok for compering with linear regression
             'n_estimators': [100,200,300,400],
             'max_depth': [3,4,5,6],
             'min_samples_leaf':[1, 3, 10],
             'learning_rate':[0.01, 0.05, 0.1]
             }

XGBOOST_PARAMETERS = {'max_depth': 6,
              'min_child_weight': 10,
              'learning_rate': 0.5,
              'obj': 'reg:linear',
              'n_estimators': 1000,
              }