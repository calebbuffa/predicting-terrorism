"""evaluate"""

from typing import Any

import geopandas as gpd
import pandas as pd
import numpy as np
import esda
import libpysal as lp


from utils import spatial_stats

predictions = spatial_stats.evaluate_predictions(r"data/out/final_predictions")

gdf, europe = spatial_stats.create_geo_dataframes(
    predictions, "data/raw/Europe.shp"
)

jc_true, jc_false_neg, jc_false_pos = spatial_stats.calculate_join_counts(gdf)
