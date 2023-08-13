from typing import Dict, List, Optional, Tuple

import circlify as circ
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

mpl.rcParams["pdf.fonttype"] = 42


def aggregate_counts(data: pd.DataFrame, levels: List[str]):
    data_dict = {}
    for n in range(len(levels)):
        # construct a groupby dataframe to obtain counts
        columns = levels[0 : (n + 1)]
        df = data.groupby(columns)[columns[0]].count().reset_index(name="count")

        # construct a nested dict to handle children levels
        for r in df.index:
            if n == 0:  # top level
                data_dict[df.iloc[r, 0]] = {"datum": df.loc[r, "count"]}
            else:
                entry = data_dict[df.iloc[r, 0]]
                for c in range(
                    1, len(columns)
                ):  # go through nested levels to find the deepest
                    if (
                        "children" not in entry
                    ):  # create a child dict if it does not exist
                        entry["children"] = {}
                    entry = entry["children"]  # go into child dict
                    if df.iloc[r, c] in entry:  # go into child dict entry if it exists
                        entry = entry[df.iloc[r, c]]
                entry[df.iloc[r, c]] = {
                    "datum": df.loc[r, "count"]
                }  # create child entry
    return data_dict


def assign_size(
    data_dict: dict,
    data: pd.DataFrame,
    levels: List[str],
    size_column: str,
    name_column: str,
):
    df = data[levels + [size_column, name_column]]
    df = df.groupby(levels + [name_column])[size_column].sum().reset_index(name="count")
    for (
        r
    ) in (
        df.index
    ):  # find the deepest levels in data_dict and create an entry with (name, size)
        entry = data_dict[df.iloc[r, 0]]
        for c in range(1, len(levels)):
            entry = entry["children"][df.iloc[r, c]]
        if "children" not in entry:
            entry["children"] = {}
        entry["children"][df.loc[r, name_column]] = {"datum": df.loc[r, "count"]}
    return data_dict


def assign_suffix(
    data_dict: dict,
    data: pd.DataFrame,
    levels: List[str],
    suffix_column: str,
    name_column: str,
):
    df = data[levels + [suffix_column, name_column]]
    for r in df.index:  # find the deepest levels in data_dict and rename with suffix
        entry = data_dict[df.iloc[r, 0]]
        for c in range(1, len(levels)):
            entry = entry["children"][df.iloc[r, c]]
        if df.loc[r, name_column] in entry["children"]:
            entry["children"][
                f"{df.loc[r, name_column]}_{df.loc[r, suffix_column]}"
            ] = entry["children"].pop(df.loc[r, name_column])
    return data_dict


def assign_colors(
    data_dict: dict,
    data: pd.DataFrame,
    levels: List[str],
    color_column: str,
    name_column: str,
):
    df = data[levels + [color_column, name_column]]
    for r in df.index:  # find the deepest levels in data_dict and rename with color
        entry = data_dict[df.iloc[r, 0]]
        for c in range(1, len(levels)):
            entry = entry["children"][df.iloc[r, c]]
        if df.loc[r, name_column] in entry["children"]:
            entry["children"][df.loc[r, color_column]] = entry["children"].pop(
                df.loc[r, name_column]
            )
    return data_dict


def get_children_data(data_dict: dict):
    child_data = []
    for i in data_dict:  # recursively get all children data
        entry = {"id": i, "datum": data_dict[i]["datum"]}
        if "children" in data_dict[i]:
            children = get_children_data(data_dict[i]["children"])
            entry["children"] = children
        child_data.append(entry)
    return child_data


def circ_dict2data(circ_dict: dict):
    circ_data = []
    for i in circ_dict:  # convert dict to circlify list data
        entry = {"id": i, "datum": circ_dict[i]["datum"]}
        if "children" in circ_dict[i]:
            children = get_children_data(circ_dict[i]["children"])
            entry["children"] = children
        circ_data.append(entry)
    return circ_data


def draw_circles(
    circ_data: list,
    title: str = "",
    fig_size: Tuple[int, int] = (10, 10),
    filename: Optional[str] = None,
    use_colormap: Optional[str] = None,
    use_suffix: Optional[dict] = None,
    use_suffix_as_color: bool = False,
):
    circles = circ.circlify(circ_data, show_enclosure=True)

    fig, ax = plt.subplots(figsize=fig_size)
    if use_colormap:
        cmap = mpl.cm.get_cmap(use_colormap)

    ax.set_title(title)  # title
    ax.axis("off")  # remove axes

    # find axis boundaries
    lim = max(
        max(abs(circle.x) + circle.r, abs(circle.y) + circle.r) for circle in circles
    )
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)

    # 1st level:
    for circle in circles:
        if circle.level != 1:
            continue
        x, y, r = circle
        ax.add_patch(
            plt.Circle(
                (x, y),
                r,
                alpha=0.5,
                linewidth=1,
                facecolor="lightblue",
                edgecolor="black",
            )
        )

    # 2nd level:
    for circle in circles:
        if circle.level != 2:
            continue
        x, y, r = circle
        plt.annotate(circle.ex["id"], (x, y), ha="center", color="black")
        ax.add_patch(
            plt.Circle(
                (x, y),
                r,
                alpha=0.5,
                linewidth=1,
                facecolor="#69b3a2",
                edgecolor="black",
            )
        )

    # 3rd level:
    for circle in circles:
        if circle.level != 3:
            continue
        x, y, r = circle

        if use_colormap:
            if use_suffix:
                suffix = circle.ex["id"].split("_")[-1]
                color_fraction = use_suffix[suffix]
            elif use_suffix_as_color:
                suffix = circle.ex["id"].split("_")[-1]
                color_fraction = float(suffix)
            else:
                color_fraction = circle.ex["id"]
            ax.add_patch(
                plt.Circle(
                    (x, y),
                    r,
                    alpha=1,
                    linewidth=1,
                    facecolor=cmap(color_fraction),
                    edgecolor="white",
                )
            )
        else:
            ax.add_patch(
                plt.Circle(
                    (x, y),
                    r,
                    alpha=0.5,
                    linewidth=1,
                    facecolor="red",
                    edgecolor="white",
                )
            )

    # 1st level labels:
    for circle in circles:
        if circle.level != 1:
            continue
        x, y, r = circle
        label = circle.ex["id"]
        plt.annotate(
            label,
            (x, y),
            va="center",
            ha="center",
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round", pad=0.5),
        )

    if filename:  # save the figure
        fig.savefig(filename, bbox_inches="tight")


def hits_heatmap(
    sample_metadata: Dict[str, pd.DataFrame],
    x: str,
    y: str,
    count_type: str = "cells",
    filename: Optional[str] = None,
):
    """Visualize a list of sample metadata objects as a heatmap.

    Parameters
    ----------
    sample_metadata: Dict[pandas.DataFrame]
        A dict where keys are cluster names and values are pandas dataframes containing
        sample metadata for each cluster centroid.
    x: str
        x-axis label key. This corresponds to cluster name values.
    y: str
        y-axis label key. This corresponds to the dataframe column to visualize.
    count_type: {"cells", "fraction"}, default: "cells"
        Count type to color in the heatmap.
    filename: str, optional
        Filename to save the figure.

    Examples
    --------
    >>> hits_heatmap(sample_metadata, "time", "disease")
    """

    valid_count_types = {"cells", "fraction"}
    if count_type not in valid_count_types:
        raise ValueError(
            f"Unknown count_type {count_type}. Options are {valid_count_types}."
        )

    for k in sample_metadata:
        sample_metadata[k][x] = k
    df = pd.concat(sample_metadata).reset_index(drop=True)

    if count_type == "cells":
        df_m = df.groupby([x, y])["cells"].sum().unstack(level=0).fillna(0)
    else:
        df_m = df.groupby([x, y])["fraction"].mean().unstack(level=0).fillna(0)

    fig, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(
        ax=ax,
        data=df_m,
        xticklabels=True,
        yticklabels=True,
        square=True,
        cmap="Blues",
        linewidth=0.3,
        cbar_kws={"shrink": 0.5},
    )
    plt.tick_params(axis="both", labelsize=4)

    # xticks
    ax.xaxis.tick_top()
    plt.xticks(np.arange(len(sample_metadata)) + 0.5, rotation=90)
    # axis labels
    plt.xlabel("")
    plt.ylabel("")
    # cbar font
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=6)

    if filename:  # save the figure
        fig.savefig(filename, bbox_inches="tight")
