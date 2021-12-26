"""spatial statistics helpers"""

from typing import Any

import geopandas as gpd
import pandas as pd
import numpy as np
import esda
import libpysal as lp


def evaluate_predictions(
    data_path: str = "../data/out/final_predictions",
) -> pd.DataFrame:
    """
    [summary]

    Parameters
    ----------
    data_path : str, optional
        [description], by default "data/out/final_predictions"

    Returns
    -------
    pd.DataFrame
        [description]
    """

    pred = pd.read_feather(data_path)

    pred["True"] = np.where(pred["Attack"] == pred["Prediction"], 1, 0)
    pred["false_neg"] = (pred["Attack"] == 1) & (pred["Prediction"] == 0)
    pred["false_pos"] = (pred["Attack"] == 0) & (pred["Prediction"] == 1)
    pred["false_neg"] = pred["false_neg"].replace({True: 1, False: 0})
    pred["false_pos"] = pred["false_pos"].replace({True: 1, False: 0})

    return pred


def create_geo_dataframes(
    predictions: pd.DataFrame, euro_file_path: str, hex_file_path: str
) -> Any:
    gdf = gpd.GeoDataFrame(
        predictions,
        geometry=gpd.points_from_xy(predictions.POINT_X, predictions.POINT_Y),
    ).set_crs("EPSG:4326")

    europe = gpd.read_file(euro_file_path)

    hex_grids = gpd.read_file(hex_file_path)

    return gdf, europe, hex_grids


def calculate_join_counts(gdf: gpd.GeoDataFrame) -> Any:
    """
    [summary]

    Returns
    -------
    Any
        [description]
    """

    wq_true = lp.weights.Queen.from_dataframe(gdf)
    wq_true.transform = "b"
    jc_true = esda.join_counts.Join_Counts(gdf["True"], wq_true)

    wq_false_neg = lp.weights.Queen.from_dataframe(gdf)
    wq_false_neg.transform = "b"
    jc_false_neg = esda.join_counts.Join_Counts(gdf["false_neg"], wq_false_neg)

    wq_false_pos = lp.weights.Queen.from_dataframe(gdf)
    wq_false_pos.transform = "b"
    jc_false_pos = esda.join_counts.Join_Counts(gdf["false_pos"], wq_false_pos)

    return jc_true, jc_false_neg, jc_false_pos
