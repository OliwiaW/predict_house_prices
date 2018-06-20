"""
first module, preparing our dataset to train model, using visualisation module
"""

import datetime

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold

import visualisations
from settings import FILE_PATH, MAX_PRICE_THERESHOLD, COLUMNS_TO_KEEP

"""
• 'id':str – database identifier of the current row,
• 'date':str – pricing date,
• 'price':float – the price
• 'bedrooms':float – number of bedrooms,
• 'bathrooms':float – number of bathrooms,
• 'sqft_living':float – living space area,
• 'sqft_lot':int – lot area,
• 'floors':float – number of floors,
• 'waterfront':int – indicator of whether the property is facing water {0,1},
• 'view':int – quality of view from the property (0:4),
• 'condition':int – property condition (1:5),
• 'grade':int – property grade (1:13),
• 'sqft_above':int – living area above ground level,
• 'sqft_basement':int – area of basement,
• 'yr_built':int – year the building was built
"""


def clean_data_format(date):
    if isinstance(date, float):  # just for case if there will be nan
        return np.NaN

    date = date.split('T')[0]
    return date[:4] + '-' + date[4:6] + '-' + date[6:]


def drop_outliers(df, col_name, max_value):
    return df[df[col_name] <= max_value]


def prepare_data():
    df = pd.read_csv(FILE_PATH)
    # first inside on data
    # print(df.describe())

    # let's clean data a little bit
    df['date'] = [clean_data_format(x) for x in df['date']]
    df['yr_built'] = [datetime.datetime.now().year - x for x in df['yr_built']]
    df['yr_renovated'] = [datetime.datetime.now().year - x if x != 0 else x for x in df['yr_renovated']]
    df['is_renovated'] = [1 if x != 0 else 0 for x in df['yr_renovated']]

    # we have some geographical data... let's plot an average price on map so that we can see if there is any correlation
    # visualisations.plot_avg_price_on_map(df) # might take a few minutes !

    # well... for future we can think about replace geo data for some useful var like amount for citizens or district (?)
    # for now let's leave it because there is needed deeper insight and maybe some crawler in beautifulsoup or scrappy

    # due to the fact that none of geo columns will be needed in further predictions
    # (postcode and coordinates however floats, but they are actually nominal values)
    # because we want to use linear regression let's drop columns which won't be used for predictions
    df = df.drop(['zipcode', 'long', 'lat'], axis=1)

    # let's see whether we have some outliers in each variable (especially in price):
    # firstly in price:
    # visualisations.plot_distribution(df, 'price')

    # we can see that there are some single high prices
    # -> in future it's might be a good idea to create a separate group for very expensive houses but now we are creating
    # one model for all so let's drop expensive houses above chosen value
    # we can do it by cutting values which std deviation is high but let's keep it simple

    df = drop_outliers(df, 'price', MAX_PRICE_THERESHOLD)

    # we can also check how time may influence on a price (inflation?)
    # visualisations.plot_price_in_time(df)

    # we cannot observe any periodical fluctuations

    # there is a lot of variables -> to decrease amount of dimensions, let's see correlation with each other
    # visualisations.plot_matrix_correlation(df)
    # based on matrix we can choose var to keep - one with no observable correlation (assume below 5%) with price or
    # highly correlated with another one will have no meaning for our model
    df = df[COLUMNS_TO_KEEP]

    # now, let's check outliers in other significant categories:
    # for column in df.columns.values:
    #     visualisations.plot_distribution(df, column)

    # there are still some single outliers in few categories but let's leave it like this
    # -> we might check further if they will be disturbing our model, let's see our prepared df:
    #print(df.head())

    # ok, there aren't any nominal var which is not translate to numeric (f.e. by using LabelEncoder or pandas dummy)
    # now it's time to split our dataset for x and y:

    X = df.drop('price', axis=1)
    y = df['price'].tolist()

    return X, y
