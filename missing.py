import numpy as np
import matplotlib as mpl
from matplotlib import gridspec
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
import seaborn as sns
import pandas as pd
def missing_value_bar(df, figsize=(20, 10), fontsize=16, labels=None, log=False, color=(0.25, 0.25, 0.25), inline=True,
        filter=None, n=0, p=0, sort=None,title_string=None):
    """
    Plots a bar chart of data nullities by column.

    :param df: The DataFrame whose completeness is being nullity matrix mapped.
    :param log: Whether or not to display a logorithmic plot. Defaults to False (linear).
    :param filter: The filter to apply to the heatmap. Should be one of "top", "bottom", or None (default). See
    `nullity_filter()` for more information.
    :param n: The cap on the number of columns to include in the filtered DataFrame. See  `nullity_filter()` for
    more information.
    :param p: The cap on the percentage fill of the columns in the filtered DataFrame. See  `nullity_filter()` for
    more information.
    :param sort: The sort to apply to the heatmap. Should be one of "ascending", "descending", or None. See
    `nullity_sort()` for more information.
    :param figsize: The size of the figure to display. This is a `matplotlib` parameter. Defaults to (24,
    10).
    :param fontsize: The figure's font size. This default to 16.
    :param labels: Whether or not to display the column names. Would need to be turned off on particularly large
    displays. Defaults to True.
    :param color: The color of the filled columns. Default is a medium dark gray: the RGB multiple `(0.25, 0.25, 0.25)`.
    :return: If `inline` is True, the underlying `matplotlib.figure` object. Else, nothing.
    """
    # Get counts.
    nullity_counts = df.isnull().sum()
    # Create the basic plot.
    fig = plt.figure(figsize=figsize)
    (nullity_counts / len(df)).plot(kind='bar', figsize=figsize, fontsize=fontsize, color=color, log=log)
    plt.title(title_string, y=1.1,fontsize = 25)

    # Get current axis.
    ax1 = plt.gca()

    # Start appending elements, starting with a modified bottom x axis.
    if labels or (labels is None and len(df.columns) <= 50):
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=fontsize)

    # Create the third axis, which displays columnar totals above the rest of the plot.
    ax3 = ax1.twiny()
    ax3.set_xticks(ax1.get_xticks())
    ax3.set_xlim(ax1.get_xlim())
    ax3.set_xticklabels(nullity_counts.values, fontsize=fontsize, rotation=45, ha='left')
    ax3.grid(False)

    # Display.
    if inline:
        plt.show()
    else:
        return fig