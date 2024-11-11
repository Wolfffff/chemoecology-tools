"""Chemoecology tools for chemical ecology analysis."""

from chemoecology_tools.analysis import perform_nmds
from chemoecology_tools.core import GCMSExperiment
from chemoecology_tools.visualization import plot_nmds
from chemoecology_tools.visualization import setup_plotting_style


__version__ = "0.1.0"

__all__ = [
    "GCMSExperiment",
    "perform_nmds",
    "plot_nmds",
    "setup_plotting_style",
]
