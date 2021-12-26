"""train models"""

# %%
import os
import warnings

import pandas as pd

from utils import preprocess

warnings.filterwarnings("ignore")

pd.options.mode.use_inf_as_na = True

# %%
data = pd.read_feather(r"/home/calebbuffa/py/thesis/data/processed/x_train")
# %%

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(root_dir)

preprocess.clean_data(
    os.path.join(root_dir, r"data/raw/training_data.csv"),
    os.path.join(root_dir, r"data/processed"),
)


# scores = utils.create_scores(
#     dnn_report, nn_report, rf_report, sgd_report, svc_report, data_out_path
# )
# # %%
# x_test.rename(
#     columns={
#         "dist_inland_water": "Distance to Inland Water",
#         "dist_majroad": "Distance to Major Roadway",
#         "dist_majroad_int": "Distance to Major Road Intersection",
#         "dist_maj_waterway": "Distance to Major Waterway",
#         "elevation": "Elevation",
#         "civil_unrest": "Civil Unrest",
#         "pop_density": "Population Density",
#         "slope": "Slope",
#         "nighttime_lights": "Nighttime Lights",
#         "urban": "Urban",
#         "agriculture": "Agriculture",
#     },
#     inplace=True,
# )

# feature_importance = rf.base_estimator_.feature_importances_
# indices = np.argsort(feature_importance)[::-1]
# labels = x_test.columns

# for f in range(x_test.shape[1]):
#     print(
#         "%2d) %-*s %f"
#         % (f + 1, 30, labels[indices[f]], feature_importance[indices[f]])
#     )
# # %%
# unsorted_list = [
#     (feature_importance, labels)
#     for labels, feature_importance in zip(labels, feature_importance)
# ]
# sorted_list = sorted(unsorted_list)

# features_sorted = []
# importance_sorted = []

# for i in sorted_list:
#     features_sorted += [i[1]]
#     importance_sorted += [i[0]]

# fp = (
#     pd.DataFrame(importance_sorted, features_sorted, columns=["Score"])
#     .reset_index()
#     .rename(columns={"index": "Feature"}, inplace=True)
# )

# plot.feature_importance(
#     "Random Forest",
#     fp["Feature"].tolist(),
#     fp["Score"].tolist(),
#     data_out_path,
#     None,
# )
