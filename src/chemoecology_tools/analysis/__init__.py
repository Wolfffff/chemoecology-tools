"""Analysis package providing statistical and dimensionality reduction methods."""

from .analysis import calculate_compositional_stats
from .analysis import perform_lda
from .analysis import perform_nmds
from .analysis import perform_pca
from .analysis import perform_random_forest
from .stats import calculate_enrichment_table


__all__ = [
    "perform_nmds",
    "calculate_enrichment_table",
    "perform_lda",
    "perform_pca",
    "perform_random_forest",
    "calculate_compositional_stats",
]
