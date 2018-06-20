"""
this module contains functions used to plotting data
"""

import folium
from bokeh.palettes import Spectral9
import branca.colormap as cm
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd


def plot_distribution(df, col_name):
    plt.hist(df[col_name])
    df['counter'] = 1
    df = df.groupby(col_name).count().reset_index()
    plt.scatter(df[col_name], df['counter'])
    plt.title(col_name)
    plt.show()


def plot_avg_price_on_map(df):
    df['lat'] = [round(x, 2) for x in df['lat']]
    df['long'] = [round(x, 2) for x in df['long']]
    df = df.groupby(by=['lat', 'long'])['price'].mean().reset_index()
    print(len(df))

    map_template = folium.Map(tiles='cartodbpositron', location=[47, -122], zoom_start=6)

    color = list(Spectral9)
    colormap = cm.StepColormap(color, vmin=df["price"].min(), vmax=df["price"].max())

    diff = df["price"].max() - df["price"].min()
    range_diff = int(diff / len(color)) + 1

    for row_number, row in df.iterrows():
        n_color = int(row['price'] / range_diff)
        if n_color > 8:
            n_color = 8
        n_size = row['price'] / 1000000
        if n_size < 2:
            n_size = 1

        folium.CircleMarker([row['lat'], row['long']],
                            radius=n_size,
                            popup="average price: " + str(round(row['price'])),
                            color=color[n_color],
                            fill_color=color[n_color]
                            ).add_to(map_template)
    map_template.add_child(colormap)
    f = folium.Figure()
    string = "Average home price"
    f.html.add_child(folium.Element("<font face='helvetica'><font size='5'><b><p1>" + string + "</b></font></p1>"))
    f.add_child(map_template)
    map_template.save('average_home_price_map.html')


def plot_price_in_time(df):
    df['year'] = [x.split('-')[0] for x in df['date']]
    df['month'] = [x.split('-')[1] for x in df['date']]
    df_detail = df.groupby('date')['price'].mean()
    df_detail.index = pd.DatetimeIndex(data=df_detail.index)
    plt.plot(df_detail, color='g')
    plt.show()
    sns.boxplot(x='year', y='price', data=df)
    plt.show()
    sns.boxplot(x='month', y='price', data=df)
    plt.show()


def plot_matrix_correlation(df):
    f, ax = plt.subplots(figsize=(14, 14))
    sns.heatmap(df.corr(), annot=True, linewidths=.5, ax=ax)
    plt.show()


def plot_score(models, scores, score_name):
    plt.xlabel('models')
    plt.ylabel('score')
    plt.title(f'{score_name} scores for various models')
    sns.barplot(models, scores)
    plt.show()
