"""utils"""

from typing import Tuple, Optional

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import inspection
from sklearn.model_selection import train_test_split


def process_landcover(data: pd.DataFrame) -> pd.DataFrame:
    """
    [summary]

    Parameters
    ----------
    data : pd.DataFrame
        [description]

    Returns
    -------
    pd.DataFrame
        [description]
    """

    data["urban"] = np.where(data["landcover"] == 50.0, 1, 0)

    data["forest"] = np.where(
        (data["landcover"] == 111.0)
        | (data["landcover"] == 112.0)
        | (data["landcover"] == 113.0)
        | (data["landcover"] == 114.0)
        | (data["landcover"] == 115.0)
        | (data["landcover"] == 116.0)
        | (data["landcover"] == 121.0)
        | (data["landcover"] == 122.0)
        | (data["landcover"] == 123.0)
        | (data["landcover"] == 124.0)
        | (data["landcover"] == 125.0)
        | (data["landcover"] == 126.0),
        1,
        0,
    )
    data["vegetation"] = np.where(
        (data["landcover"] == 20.0)
        | (data["landcover"] == 30.0)
        | (data["landcover"] == 90.0)
        | (data["landcover"] == 100.0),
        1,
        0,
    )
    data["agriculture"] = np.where((data["landcover"] == 40.0), 1, 0)
    data["snow/ice/water"] = np.where(
        (data["landcover"] == 70.0)
        | (data["landcover"] == 80.0)
        | (data["landcover"] == 200.0),
        1,
        0,
    )
    data.drop(columns="landcover", inplace=True)

    return data


def feature_selection(
    y_train: pd.DataFrame,
    y_eval: pd.DataFrame,
    x_train: pd.DataFrame,
    x_eval: pd.DataFrame,
    x_test: pd.DataFrame,
    verbose: Optional[bool]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    [summary]

    Parameters
    ----------
    y_train : pd.DataFrame
        [description]
    y_eval : pd.DataFrame
        [description]
    x_train : pd.DataFrame
        [description]
    x_eval : pd.DataFrame
        [description]
    x_test : pd.DataFrame
        [description]

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        [description]
    """

    feature_selection_model = RandomForestClassifier(
        n_estimators=100, n_jobs=-1, random_state=1, class_weight="balanced"
    )

    feature_selection_model.fit(x_train, y_train)

    pi = inspection.permutation_importance(
        feature_selection_model, x_eval, y_eval, n_repeats=30, random_state=1
    )

    if verbose:
        print("Permutation Importance Scores:")
        for i in pi.importances_mean.argsort()[::-1]:
            print(f"{x_train.columns[i]} : {pi.importances_mean[i]:.3f}")

    x_train.drop(
        [
            "f_15_49",
            "pop_count",
            "built_settlement_growth",
            "snow/ice/water",
            "vegetation",
            "gdp",
            "impervious_land",
            "forest",
        ],
        inplace=True,
        axis=1,
    )
    x_eval.drop(
        [
            "f_15_49",
            "pop_count",
            "built_settlement_growth",
            "snow/ice/water",
            "vegetation",
            "gdp",
            "impervious_land",
            "forest",
        ],
        inplace=True,
        axis=1,
    )
    x_test.drop(
        [
            "f_15_49",
            "pop_count",
            "built_settlement_growth",
            "snow/ice/water",
            "vegetation",
            "gdp",
            "impervious_land",
            "forest",
        ],
        inplace=True,
        axis=1,
    )

    return x_train, x_eval, x_test


# %%
def clean_data(data_path: str, out_path: str) -> None:
    """
    [summary]

    Parameters
    ----------
    data_path : str
        [description]
    out_path : str
        [description]
    """

    orig_data = pd.read_csv(data_path, index_col=False)

    data = orig_data[
        [
            "OID_",
            "POINT_X",
            "POINT_Y",
            "Attack",
            "built_settlement_growth",
            "landcover",
            "gdp",
            "dist_inland_water",
            "dist_majroad",
            "dist_majroad_int",
            "dist_maj_waterway",
            "impervious_land",
            "elevation",
            "civil_unrest",
            "f_15_49",
            "pop_density",
            "slope",
            "nighttime_lights",
            "pop_count",
        ]
    ]

    data["civil_unrest"] = data["civil_unrest"].fillna(0)

    data = process_landcover(data)

    data.dropna(axis=0, how="any", inplace=True)

    data.reset_index().iloc[:, 1:].to_feather(
        os.path.join(out_path, "processed_training_data")
    )


def prepare_training_data(
    data_path: pd.DataFrame, verbose: Optional[bool]
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    [summary]

    Parameters
    ----------
    data : pd.DataFrame
        [description]
    out_path : str
        [description]

    Returns
    -------
    Tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame,
        pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
        [description]
    """

    data = pd.read_feather(data_path)

    x = data.iloc[:, 4:]
    y = data.iloc[:, 3]

    x_train, x_other, y_train, y_other = train_test_split(
        x, y, test_size=0.3, stratify=y, random_state=1
    )
    x_eval, x_test, y_eval, y_test = train_test_split(
        x_other, y_other, test_size=0.5, stratify=y_other, random_state=1
    )

    x_train, x_eval, x_test = feature_selection(
        y_train, y_eval, x_train, x_eval, x_test, verbose
    )

    x_train.dropna(axis=0, how="any", inplace=True)
    x_eval.dropna(axis=0, how="any", inplace=True)
    x_test.dropna(axis=0, how="any", inplace=True)

    return (
        x_train,
        x_eval,
        x_test,
        y_train,
        y_eval,
        y_test,
        data[["POINT_X", "POINT_Y"]],
    )


def create_scores(
    dnn_report: pd.DataFrame,
    nn_report: pd.DataFrame,
    rf_report: pd.DataFrame,
    sgd_report: pd.DataFrame,
    svc_report: pd.DataFrame,
    out_path: str,
) -> pd.DataFrame:
    """
    [summary]

    Parameters
    ----------
    dnn_report : pd.DataFrame
        [description]
    nn_report : pd.DataFrame
        [description]
    rf_report : pd.DataFrame
        [description]
    sgd_report : pd.DataFrame
        [description]
    svc_report : pd.DataFrame
        [description]
    out_path : str
        [description]

    Returns
    -------
    pd.DataFrame
        [description]
    """

    accuracy_scores = {
        "DNN": dnn_report.loc["accuracy"][0],
        "NN": nn_report.loc["accuracy"][0],
        "Random Forest": rf_report.loc["accuracy"][0],
        "SGD Log Reg": sgd_report.loc["accuracy"][0],
        "Ensemble SVM": svc_report.loc["accuracy"][0],
    }

    precision_scores = {
        "DNN": dnn_report.iloc[3][0],
        "NN": nn_report.iloc[3][0],
        "Random Forest": rf_report.iloc[3][0],
        "SGD Log Reg": sgd_report.iloc[3][0],
        "Ensemble SVM": svc_report.iloc[3][0],
    }
    recall_scores = {
        "DNN": dnn_report.iloc[3][1],
        "NN": nn_report.iloc[3][1],
        "Random Forest": rf_report.iloc[3][1],
        "SGD Log Reg": sgd_report.iloc[3][1],
        "Ensemble SVM": svc_report.iloc[3][1],
    }
    f1_scores = {
        "DNN": dnn_report.iloc[3][2],
        "NN": nn_report.iloc[3][2],
        "Random Forest": rf_report.iloc[3][2],
        "SGD Log Reg": sgd_report.iloc[3][2],
        "Ensemble SVM": svc_report.iloc[3][2],
    }

    accuracy = pd.DataFrame.from_dict(accuracy_scores, orient="index")
    precision = pd.DataFrame.from_dict(precision_scores, orient="index")
    recall = pd.DataFrame.from_dict(recall_scores, orient="index")
    f1 = pd.DataFrame.from_dict(f1_scores, orient="index")

    scores = pd.concat([accuracy, precision, recall, f1], axis=1)
    scores.columns = ["Accuracy", "Precision", "Recall", "F1"]

    scores.sort_values(by="F1", ascending=False)
    scores.to_csv(os.path.join(out_path, "model_scores.csv"), index=False)

    return scores
