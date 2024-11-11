"""Core analysis functions for dimensionality reduction and pattern detection."""

import pandas as pd
from sklearn.manifold import MDS

from ..core.gcms_experiment import GCMSExperiment


def perform_nmds(
    experiment: GCMSExperiment, n_components: int = 2, random_state: int = 42
) -> pd.DataFrame:
    """Perform NMDS on chemical data.

    Args:
        experiment: GCMSExperiment instance containing the data
        n_components: Number of dimensions to reduce to
        random_state: Random seed for reproducibility

    Returns:
        DataFrame containing NMDS coordinates (NMDS1, NMDS2)
    """
    mds = MDS(
        n_components=n_components,
        dissimilarity="euclidean",
        random_state=random_state,
    )
    nmds_coords = mds.fit_transform(experiment.get_abundance_matrix())

    return pd.DataFrame(
        nmds_coords, columns=[f"NMDS{i+1}" for i in range(n_components)]
    )
