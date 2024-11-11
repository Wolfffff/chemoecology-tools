"""Module for creating beautiful plots using Matplotlib and Seaborn."""

from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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
) -> tuple[plt.Figure, plt.Axes]:
    """Create a new figure with specified dimensions.

    Args:
        width: Figure width in inches
        aspect_ratio: Height/width ratio

    Returns:
        tuple of (Figure, Axes) objects
    """
    height = width * aspect_ratio
    fig, ax = plt.subplots(figsize=(width, height))
    return fig, ax


def style_nmds_plot(ax: plt.Axes, title: str, legend_title: str | None = None) -> None:
    """Apply consistent styling to NMDS plots.

    Args:
        ax: matplotlib axes object to style
        title: Plot title
        legend_title: Optional title for the legend
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
) -> plt.Figure:
    """Create a beautifully styled NMDS plot for GCMS experiment data.

    Args:
        experiment: GCMSExperiment instance containing the data
        nmds_coords: DataFrame containing NMDS coordinates (NMDS1, NMDS2)
        group_col: Optional metadata column name to group/color points by
        title: Plot title
        width: Figure width in inches
        aspect_ratio: Height/width ratio for the figure

    Returns:
        matplotlib Figure object containing the styled plot
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
