"""train models"""

import argparse

import os
import json

import pickle
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    BaggingClassifier,
    RandomForestClassifier,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from utils.preprocess import prepare_training_data


def setup_args() -> argparse.ArgumentParser:
    """
    [summary]

    Returns
    -------
    argparse.ArgumentParser
        [description]
    """

    parser = argparse.ArgumentParser(
        description="train terrorism prediction models"
    )

    parser.add_argument("-gs", "--grid_search", action="store_true")

    parser.add_argument(
        "-o", "--out_path", required=False, type=str, default="../data/out"
    )

    parser.add_argument(
        "-i",
        "--input_data",
        required=False,
        type=str,
        default="../data/processed/processed_training_data",
    )

    return parser


# ROOT_DIR = os.path.basename(os.path.abspath(__file__))
# DATA_DIR = os.path.join(ROOT_DIR, "data")


class ExperimentPipeline:
    """
    [summary]
    """

    def __init__(self, in_path: str, out_path: str) -> None:
        self.data_path = in_path
        self.out_path = out_path

        if (
            not os.path.exists("../data/processed/x_train")
            or not os.path.exists("../data/processed/y_train")
            or not os.path.exists("../data/processed/x_eval")
            or not os.path.exists("../data/processed/y_eval")
            or not os.path.exists("../data/processed/x_test")
            or not os.path.exists("../data/processed/y_test")
        ):
            print("Preparing data...")
            (
                self.x_train,
                self.x_eval,
                self.x_test,
                self.y_train,
                self.y_eval,
                self.y_test,
                self.data,
            ) = prepare_training_data(self.data_path, False)

            self.x_train.reset_index().iloc[:, 1:].to_feather(
                "../data/processed/x_train"
            )

            self.y_train.reset_index().iloc[:, 1:].to_feather(
                "../data/processed/y_train"
            )

            self.x_eval.reset_index().iloc[:, 1:].to_feather(
                "../data/processed/x_eval"
            )

            self.y_eval.reset_index().iloc[:, 1:].to_feather(
                "../data/processed/y_eval"
            )

            self.x_test.reset_index().iloc[:, 1:].to_feather(
                "../data/processed/x_test"
            )

            self.y_test.reset_index().iloc[:, 1:].to_feather(
                "../data/processed/y_test"
            )

            print("Data prepared!")

        else:

            print("Reading in cached data...")
            self.x_train = pd.read_feather("../data/processed/x_train")
            self.y_train = pd.read_feather("../data/processed/y_train")
            self.x_eval = pd.read_feather("../data/processed/x_eval")
            self.y_eval = pd.read_feather("../data/processed/y_eval")
            self.x_test = pd.read_feather("../data/processed/x_test")
            self.y_test = pd.read_feather("../data/processed/y_test")
            self.data = pd.read_feather(
                self.data_path, columns=["POINT_X", "POINT_Y"]
            )

            print("Data read!")

    def random_forest(self, grid_search: bool) -> None:
        """
        [summary]

        Parameters
        ----------
        grid_search : bool
            [description]
        """

        rf_est = RandomForestClassifier(
            class_weight="balanced", verbose=3, random_state=0
        )

        if grid_search:

            n_estimators = [100, 200, 300]
            criterion = ["gini", "entropy"]
            max_depth = [5, 10, 15, 20, 40, 80, 100, "None"]
            min_samples_split = [2, 4, 6, 8, 10]
            max_features = ["auto", "sqrt", None]
            min_samples_leaf = [1, 2, 4]
            max_samples = [0.25, 0.5, 0.75, 1.0]

            param_grid = {
                "n_estimators": n_estimators,
                "criterion": criterion,
                "max_depth": max_depth,
                "min_samples_split": min_samples_split,
                "max_features": max_features,
                "min_samples_leaf": min_samples_leaf,
                "max_samples": max_samples,
            }

            rf = RandomizedSearchCV(
                estimator=rf_est,
                param_distributions=param_grid,
                n_jobs=-1,
                cv=3,
                verbose=4,
                random_state=1,
            )

        else:
            rf = RandomForestClassifier(
                class_weight="balanced_subsample", random_state=1
            )

        model = make_pipeline(MinMaxScaler(), rf)

        model.fit(self.x_train, self.y_train.to_numpy().ravel())

        rf_pred = model.predict(self.x_eval)

        rf_report = classification_report(
            self.y_eval, rf_pred, output_dict=True
        )

        with open(
            os.path.join(self.out_path, "random_forest_report.json"), "w"
        ) as out_file:
            json.dump(rf_report, out_file, indent=4)

        predictions = pd.DataFrame(self.y_eval).copy()

        predictions["pred"] = rf_pred
        predictions = predictions.join(
            self.data, on=predictions.index, how="left"
        )

        predictions.to_feather(
            os.path.join(self.out_path, "random_forest_predictions")
        )

        model_out = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "models",
            "random_forest.sav",
        )

        with open(model_out, "wb") as out:
            pickle.dump(model, out)

    def log_reg(self, grid_search: bool,) -> None:
        """
        [summary]

        Parameters
        ----------
        grid_search : bool
            [description]
        """

        if grid_search:

            sgd_est = SGDClassifier(
                penalty="elasticnet",
                class_weight="balanced",
                early_stopping=True,
            )

            loss = ["log", "hinge"]
            l1_ratio = np.linspace(0.0, 1.0, 5)
            learning_rate = ["invscaling", "adaptive", "constant", "optimal"]
            eta0 = np.linspace(0.1, 1.0, 5)
            power_t = [0.1, 0.25, 0.5, 0.75, 1.0]
            alpha = 10.0 ** -np.arange(1, 7)
            param_grid = {
                "loss": loss,
                "l1_ratio": l1_ratio,
                "learning_rate": learning_rate,
                "eta0": eta0,
                "power_t": power_t,
                "alpha": alpha,
            }

            sgd = RandomizedSearchCV(
                estimator=sgd_est,
                param_distributions=param_grid,
                n_jobs=-1,
                cv=3,
                verbose=4,
                random_state=1,
                n_iter=5,
            )

        else:
            sgd = SGDClassifier(
                alpha=1e-06,
                class_weight="balanced",
                early_stopping=True,
                eta0=0.1,
                l1_ratio=1.0,
                loss="log",
                penalty="elasticnet",
            )

        model = make_pipeline(MinMaxScaler(), sgd)

        model.fit(self.x_train, self.y_train.to_numpy().ravel())

        sgd_pred = model.predict(self.x_eval)

        sgd_report = classification_report(
            self.y_eval, sgd_pred, output_dict=True
        )

        with open(
            os.path.join(self.out_path, "log_reg_report.json"), "w"
        ) as out_file:
            json.dump(sgd_report, out_file, indent=4)

        predictions = pd.DataFrame(self.y_eval).copy()

        predictions["pred"] = sgd_pred
        predictions = predictions.join(
            self.data, on=predictions.index, how="left"
        )

        predictions.to_feather(
            os.path.join(self.out_path, "log_reg_predictions")
        )

        model_out = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "models",
            "logistic_regression.sav",
        )

        with open(model_out, "wb") as out:
            pickle.dump(model, out)

    def svm(self, grid_search: bool,) -> None:
        """
        [summary]

        Parameters
        ----------
        grid_search : bool
            [description]
        """

        n_estimators = 10

        if grid_search:

            param_grid = dict(
                base_estimator__C=np.logspace(-2, 10, 13),
                base_estimator__gamma=np.logspace(-9, 3, 13),
            )

            svc = RandomizedSearchCV(
                BaggingClassifier(
                    SVC(
                        kernel="rbf",
                        class_weight="balanced",
                        probability=True,
                        random_state=0,
                        verbose=True,
                    ),
                    max_samples=1.0 / n_estimators,
                    n_estimators=n_estimators,
                    random_state=0,
                    bootstrap=False,
                ),
                param_distributions=param_grid,
                cv=3,
                verbose=4,
                n_jobs=-1,
                n_iter=5,
            )

        else:
            svc = BaggingClassifier(
                SVC(kernel="rbf", class_weight="balanced", random_state=0),
                max_samples=1.0 / n_estimators,
                n_estimators=n_estimators,
                random_state=0,
                bootstrap=False,
            )

        model = make_pipeline(MinMaxScaler(), svc)

        model.fit(self.x_train, self.y_train.to_numpy().ravel())

        svc_pred = model.predict(self.x_eval)

        svc_report = classification_report(
            self.y_eval, svc_pred, output_dict=True
        )

        with open(
            os.path.join(self.out_path, "svm_report.json"), "w"
        ) as out_file:
            json.dump(svc_report, out_file, indent=4)

        predictions = pd.DataFrame(self.y_eval).copy()

        predictions["pred"] = svc_pred
        predictions = predictions.join(
            self.data, on=predictions.index, how="left"
        )

        predictions.to_feather(os.path.join(self.out_path, "svm_predictions"))

        model_out = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "models",
            "svm.sav",
        )

        with open(model_out, "wb") as out:
            pickle.dump(model, out)

    def neural_network(self) -> None:
        """
        [summary]
        """

        nn = MLPClassifier(
            hidden_layer_sizes=(10),
            activation="logistic",
            solver="adam",
            learning_rate_init=0.001,
            max_iter=500,
            random_state=1,
        )

        model = make_pipeline(MinMaxScaler(), nn)

        model.fit(self.x_train, self.y_train.to_numpy().ravel())

        nn_pred = model.predict(self.x_eval)

        nn_report = classification_report(
            self.y_eval, nn_pred, output_dict=True
        )

        with open(
            os.path.join(self.out_path, "nn_report.json"), "w"
        ) as out_file:
            json.dump(nn_report, out_file, indent=4)

        predictions = pd.DataFrame(self.y_eval).copy()

        predictions["pred"] = nn_pred
        predictions = predictions.join(
            self.data, on=predictions.index, how="left"
        )

        predictions.to_feather(os.path.join(self.out_path, "nn_predictions"))

        model_out = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "models",
            "nn.sav",
        )

        with open(model_out, "wb") as out:
            pickle.dump(model, out)

    def deep_neural_network(self) -> None:
        """
        [summary]
        """

        dnn = MLPClassifier(
            hidden_layer_sizes=(100, 50, 30, 10, 5),
            activation="logistic",
            solver="adam",
            learning_rate_init=0.001,
            max_iter=500,
            random_state=1,
        )

        model = make_pipeline(MinMaxScaler(), dnn)

        model.fit(self.x_train, self.y_train.to_numpy().ravel())

        dnn_pred = model.predict(self.x_eval)

        predictions = pd.DataFrame(self.y_eval).copy()

        dnn_report = classification_report(
            self.y_eval, dnn_pred, output_dict=True
        )

        with open(
            os.path.join(self.out_path, "dnn_report.json"), "w"
        ) as out_file:
            json.dump(dnn_report, out_file, indent=4)

        predictions["pred"] = dnn_pred
        predictions = predictions.join(
            self.data, on=predictions.index, how="left"
        )

        predictions.to_feather(os.path.join(self.out_path, "dnn_predictions"))

        model_out = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "models",
            "dnn.sav",
        )

        with open(model_out, "wb") as out:
            pickle.dump(model, out)


if __name__ == "__main__":

    argument_parser = setup_args()

    args = argument_parser.parse_args()

    experiment = ExperimentPipeline(args.input_data, args.out_path)

    print("training random forest..")
    experiment.random_forest(args.grid_search)

    print("training svm...")
    experiment.svm(args.grid_search)

    print("logistic regression...")
    experiment.log_reg(args.grid_search)

    print("neural network...")
    experiment.neural_network()

    print("deep neural network...")
    experiment.deep_neural_network()
