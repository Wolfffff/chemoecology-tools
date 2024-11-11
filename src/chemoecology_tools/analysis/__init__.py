"""Analysis package providing statistical and dimensionality reduction methods."""

from .analysis import perform_nmds
from .stats import calculate_enrichment_table


__all__ = ["perform_nmds", "calculate_enrichment_table"]
