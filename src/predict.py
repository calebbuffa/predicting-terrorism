"""predict data"""

import argparse
import pickle
import os
import pandas as pd


def setup_args() -> argparse.ArgumentParser:
    """
    [summary]
    """

    parser = argparse.ArgumentParser(description="Predict on all data")

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        choices=["rf", "svm", "lr", "nn", "dnn"],
        default="rf",
        required=False,
    )
    parser.add_argument(
        "-d",
        "--data_path",
        type=str,
        default="../data/processed/processed_training_data",
        required=False,
    )
    parser.add_argument(
        "-o", "--out_path", type=str, default="../data/out/", required=False
    )

    return parser


def predict() -> None:
    """predict on test data"""

    parser = setup_args()
    args = parser.parse_args()

    model_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "models"
    )

    model_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "models"
    )

    if args.model == "rf":
        with open(
            os.path.join(model_path, "random_forest.sav"), "rb"
        ) as model:
            clf = pickle.load(model)

    elif args.model == "lr":
        with open(
            os.path.join(model_path, "logistic_regression.sav"), "rb"
        ) as model:
            clf = pickle.load(model)

    elif args.model == "svm":
        with open(os.path.join(model_path, "svm.sav"), "rb") as model:
            clf = pickle.load(model)

    elif args.model == "nn":
        with open(os.path.join(model_path, "nn.sav"), "rb") as model:
            clf = pickle.load(model)

    elif args.model == "dnn":
        with open(os.path.join(model_path, "dnn.sav"), "rb") as model:
            clf = pickle.load(model)

    else:
        clf = None
        raise argparse.ArgumentError

    data = pd.read_feather(args.data_path)

    x = data[
        [
            "dist_inland_water",
            "dist_majroad",
            "dist_majroad_int",
            "dist_maj_waterway",
            "elevation",
            "civil_unrest",
            "pop_density",
            "slope",
            "nighttime_lights",
            "urban",
            "agriculture",
        ]
    ]

    y_hat = clf.predict(x)

    joined = pd.concat(
        [data, pd.DataFrame(y_hat, columns=["Prediction"])], axis=1
    )

    joined.to_feather(os.path.join(args.out_path, "final_predictions"))


if __name__ == "__main__":

    predict()
