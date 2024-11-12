"""Tests for the GCMSExperiment class."""

import pandas as pd
from pandas import DataFrame

from chemoecology_tools.core import GCMSExperiment


def test_basic_gcms_experiment() -> None:
    """Test basic properties of the GCMSExperiment class."""
    # Create minimal test data with explicit types
    abundance_df: DataFrame = pd.DataFrame(
        {"ID": ["S1", "S2"], "Chemical1": [0.5, 0.3], "Chemical2": [0.5, 0.7]}
    )

    metadata_df: DataFrame = pd.DataFrame({"ID": ["S1", "S2"], "Type": ["A", "B"]})

    # Create experiment
    exp: GCMSExperiment = GCMSExperiment(abundance_df, metadata_df)

    # Test basic properties
    assert len(exp) == 2  # Number of samples
    assert len(exp.chemical_cols) == 2  # Number of chemicals
    assert exp.id_col == "ID"  # Default ID column
    assert exp.chemical_cols == ["Chemical1", "Chemical2"]
