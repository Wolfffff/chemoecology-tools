"""Module for creating beautiful plots using Matplotlib and Seaborn."""

from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..core.gcms_experiment import GCMSExperiment


# Scientific visualization color palettes
COLOR_PALETTES: dict[str, Any] = {
    "main": ["#2274A5", "#F75C03", "#F1C40F", "#D90368", "#00CC66"],
    "categorical": sns.color_palette("deep"),
    "sequential": sns.color_palette("viridis"),
}

# Default plot settings
FONT_SETTINGS: dict[str, str | list[str] | int] = {
    "family": "sans-serif",
    "sans-serif": ["Arial"],
    "size": 12,
    "label_size": 14,
    "title_size": 16,
}

FIGURE_SETTINGS: dict[str, float | int] = {
    "dpi": 300,
    "save_dpi": 300,
    "default_width": 10,
    "golden_ratio": 0.618,
}


def setup_plotting_style() -> None:
    """Configure global matplotlib and seaborn plotting style settings."""
    plt.style.use("tableau-colorblind10")
    sns.set_style(
        "ticks",
        {
            "axes.grid": True,
            "grid.color": ".8",
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
        },
    )

    # Font configuration
    plt.rcParams["font.family"] = FONT_SETTINGS["family"]
    plt.rcParams["font.sans-serif"] = FONT_SETTINGS["sans-serif"]
    plt.rcParams["font.size"] = FONT_SETTINGS["size"]
    plt.rcParams["axes.labelsize"] = FONT_SETTINGS["label_size"]
    plt.rcParams["axes.titlesize"] = FONT_SETTINGS["title_size"]
    plt.rcParams["xtick.labelsize"] = FONT_SETTINGS["size"]
    plt.rcParams["ytick.labelsize"] = FONT_SETTINGS["size"]

    # Figure configuration
    plt.rcParams["figure.figsize"] = (
        FIGURE_SETTINGS["default_width"],
        FIGURE_SETTINGS["default_width"] * FIGURE_SETTINGS["golden_ratio"],
    )
    plt.rcParams["figure.dpi"] = FIGURE_SETTINGS["dpi"]
    plt.rcParams["savefig.dpi"] = FIGURE_SETTINGS["save_dpi"]
    plt.rcParams["figure.constrained_layout.use"] = True


def create_figure(
    width: float = FIGURE_SETTINGS["default_width"],
    aspect_ratio: float = FIGURE_SETTINGS["golden_ratio"],
) -> tuple[Figure, Axes]:
    """Create a new figure with specified dimensions.

    Args:
        width: Figure width in inches. Defaults to FIGURE_SETTINGS["default_width"].
        aspect_ratio: Height/width ratio. Defaults to FIGURE_SETTINGS["golden_ratio"].

    Returns:
        A tuple containing:
            - Figure: The matplotlib figure object
            - Axes: The matplotlib axes object
    """
    height = width * aspect_ratio
    fig, ax = plt.subplots(figsize=(width, height))
    return fig, ax


def style_nmds_plot(ax: Axes, title: str, legend_title: str | None = None) -> None:
    """Apply consistent styling to NMDS plots.

    Args:
        ax: Matplotlib axes object to style.
        title: Plot title text.
        legend_title: Optional title for the legend. Defaults to None.
    """
    ax.set_xlabel("NMDS1", fontsize=FONT_SETTINGS["label_size"], fontweight="bold")
    ax.set_ylabel("NMDS2", fontsize=FONT_SETTINGS["label_size"], fontweight="bold")
    ax.set_title(title, fontsize=FONT_SETTINGS["title_size"], pad=20)
    ax.set_aspect("equal")

    if legend_title:
        legend = ax.get_legend()
        if legend:
            legend.set_title(legend_title)
            legend.set_frame_on(True)
            legend.get_frame().set_facecolor("white")
            legend.get_frame().set_alpha(0.9)


def plot_nmds(
    experiment: GCMSExperiment,
    nmds_coords: pd.DataFrame,
    group_col: str | None = None,
    title: str = "NMDS Plot",
    width: float = FIGURE_SETTINGS["default_width"],
    aspect_ratio: float = FIGURE_SETTINGS["golden_ratio"],
) -> Figure:
    """Create a beautifully styled NMDS plot for GCMS experiment data.

    Args:
        experiment: GCMSExperiment instance containing the data.
        nmds_coords: DataFrame containing NMDS coordinates (NMDS1, NMDS2).
        group_col: Optional metadata column name to group/color points by.
            Defaults to None.
        title: Plot title. Defaults to "NMDS Plot".
        width: Figure width in inches. Defaults to
            FIGURE_SETTINGS["default_width"].
        aspect_ratio: Height/width ratio for the figure.
            Defaults to FIGURE_SETTINGS["golden_ratio"].

    Returns:
        Figure: Matplotlib Figure object containing the styled plot.
    """
    setup_plotting_style()
    fig, ax = create_figure(width, aspect_ratio)

    if group_col and group_col in experiment.metadata_df.columns:
        plot_data = pd.concat(
            [nmds_coords, experiment.metadata_df[[group_col]]], axis=1
        )
        sns.scatterplot(
            data=plot_data,
            x="NMDS1",
            y="NMDS2",
            hue=group_col,
            style=group_col,
            palette=COLOR_PALETTES["main"],
            s=100,
            alpha=0.7,
            ax=ax,
        )
        style_nmds_plot(ax, title, group_col)
    else:
        sns.scatterplot(data=nmds_coords, x="NMDS1", y="NMDS2", s=100, alpha=0.7, ax=ax)
        style_nmds_plot(ax, title)

    return fig


def plot_pca(
    pca_results: dict[str, Any],
    experiment: GCMSExperiment,
    group_col: str = "Caste",
    title: str | None = None,
    figsize: tuple[int, int] = (10, 8),
) -> Figure:
    """Plot PCA results.

    Args:
        pca_results: Dictionary containing PCA results and coordinates.
        experiment: GCMSExperiment instance containing the data.
        group_col: Column name for grouping. Defaults to "Caste".
        title: Optional plot title. Defaults to None.
        figsize: Figure dimensions (width, height). Defaults to (10, 8).

    Returns:
        Figure: Matplotlib Figure object containing the PCA plot.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Create plot data
    plot_data = pd.concat(
        [pca_results["coords"], experiment.metadata_df[[group_col]]], axis=1
    )

    # Create scatter plot
    sns.scatterplot(data=plot_data, x="PC1", y="PC2", hue=group_col, alpha=0.7, ax=ax)

    # Add variance explained
    var_explained = pca_results["explained_variance"]
    ax.set_xlabel(f"PC1 ({var_explained[0]:.1%} var. explained)")
    ax.set_ylabel(f"PC2 ({var_explained[1]:.1%} var. explained)")

    if title:
        ax.set_title(title)

    return fig


def plot_lda(
    lda_results: dict[str, Any],
    experiment: GCMSExperiment,
    group_col: str = "Caste",
    title: str | None = None,
    figsize: tuple[int, int] = (10, 8),
) -> Figure:
    """Plot LDA results.

    Args:
        lda_results: Dictionary containing LDA results and coordinates.
        experiment: GCMSExperiment instance containing the data.
        group_col: Column name for grouping. Defaults to "Caste".
        title: Optional plot title. Defaults to None.
        figsize: Figure dimensions (width, height). Defaults to (10, 8).

    Returns:
        Figure: Matplotlib Figure object containing the LDA plot.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Create plot data
    plot_data = pd.concat(
        [lda_results["coords"], experiment.metadata_df[[group_col]]], axis=1
    )

    # Create scatter plot
    sns.scatterplot(data=plot_data, x="LD1", y="LD2", hue=group_col, alpha=0.7, ax=ax)

    if title:
        ax.set_title(title)

    return fig


def plot_rf_importance(
    rf_results: dict[str, Any],
    n_features: int = 10,
    title: str | None = None,
    figsize: tuple[int, int] = (12, 6),
) -> Figure:
    """Plot Random Forest feature importance.

    Args:
        rf_results: Dictionary containing Random Forest results and feature importance.
        n_features: Number of top features to display. Defaults to 10.
        title: Optional plot title. Defaults to None.
        figsize: Figure dimensions (width, height). Defaults to (12, 6).

    Returns:
        Figure: Matplotlib Figure object containing the feature importance plot.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)

    importance_data = rf_results["feature_importance"].head(n_features)

    ax.bar(importance_data["feature"], importance_data["importance"])
    plt.xticks(rotation=45, ha="right")

    ax.set_xlabel("Chemical Compound")
    ax.set_ylabel("Feature Importance")

    if title:
        ax.set_title(title)

    plt.tight_layout()
    return fig
