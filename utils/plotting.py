import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from common.helper import map_feature_label, map_target_variable

def get_grouped_dataframe_by_column(
    dataframe: pd.DataFrame,
    column_name: str,
    percentage: bool=True,
    dropna: bool=True,
    reverse: bool=True
):

    groupby = column_name if reverse else "DIABETES"

    grouped_df = dataframe[["DIABETES", column_name]]
    grouped_df = grouped_df.groupby(by=groupby, dropna=dropna).value_counts(dropna=dropna, normalize=percentage).reset_index()

    return grouped_df

def generate_stacked_bar_plot(
    dataframe: pd.DataFrame,
    column_name: str,
    percentage: bool=True,
    labels: dict=None,
    legend_loc: str="upper left",
    ax: plt.Axes=None,
    dropna: bool=True,
    title: str=None,
    bar_width: int=0.5,
    reverse: bool=True,
    ascending: bool=True
):
    gdf = get_grouped_dataframe_by_column(dataframe, column_name,
    percentage, dropna, reverse)

    gdf = map_target_variable(gdf)

    if labels is not None:
        gdf = map_feature_label(gdf, column_name, labels, dropna)

    target_label = "DIABETES"

    x = target_label if reverse else column_name
    y = column_name if reverse else target_label

    values = gdf[x].unique()
    cnt_column = "proportion" if percentage else "count"
    weight_counts = {
        k: gdf[gdf[x] == k].sort_index(ascending=ascending)[cnt_column].to_list()
                for k in values
    }


    if ax is None:
        _, ax = plt.subplots()

    columns = gdf[y].unique()
    bottom = np.zeros(len(columns))


    for label, weight_count in weight_counts.items():
        ax.bar(
            x=columns,
            height=weight_count,
            width=bar_width,
            label=label,
            bottom=bottom
        )

        bottom += weight_count

    ax.set_title(title)
    ax.legend(loc=legend_loc)
    ax.set_xlabel("")

def generate_grouped_bar_plot(
    dataframe: pd.DataFrame,
    column_name: str,
    percentage: bool=True,
    labels: dict=None,
    ax: plt.Axes=None,
    dropna: bool=True,
    title: str=None,
    legend_loc: str="upper left",
    reverse: bool=True
):
    gdf = get_grouped_dataframe_by_column(dataframe, column_name,
    percentage, dropna, reverse)

    gdf = map_target_variable(gdf)
    
    if labels is not None:
        gdf = map_feature_label(gdf, column_name, labels, dropna)

    x = "DIABETES"
    y = column_name

    if reverse:
        x = column_name
        y = "DIABETES"

    cnt_column = "proportion" if percentage else "count"

    gdf = gdf.pivot(index=x, columns=y, values=cnt_column)

    if ax is None:
        _, ax = plt.subplots()

    ax.set_title(title)
    ax.set_xlabel("")
    gdf.plot.bar(ax=ax, rot=0)
    ax.legend(loc=legend_loc)
    