"""plotting functions"""

from __future__ import annotations

import os
from typing import Optional

import matplotlib.pyplot as plt


def feature_importance(
    model: str,
    feature_names: list[str],
    scores: list[float | int],
    out_path: str,
    ax: Optional[plt.Axes],
) -> plt.Axes:
    """
    [summary]

    Parameters
    ----------
    model : str
        [description]
    feature_names : list[str]
        [description]
    scores : pd.DataFrame
        [description]
    out_path : str
        [description]
    ax : Optional[plt.Axes]
        [description]

    Returns
    -------
    plt.Axes
        [description]
    """

    if ax is None:
        ax = plt.gca()

    ax.figure(figsize=(8, 6))

    ax.set_title(f"{model} Feature Importance", size=20)
    ax.set_ylabel("Importance Score", size=20)
    ax.set_xlabel("Variable", size=20)
    ax.grid(True, linestyle="dotted")
    ax.bar(
        feature_names, scores, color="gray",
    )

    ax.xticks(rotation=45, size=11, ha="right")
    ax.yticks(size=12)

    ax.tight_layout()
    path = os.path.join(out_path, "eature_importance.jpg")
    plt.savefig(
        path,
        dpi=300,
        facecolor="w",
        edgecolor="b",
        orientation="portrait",
        transparent=False,
        bbox_inches=None,
        pad_inches=0.1,
    )

    return ax
