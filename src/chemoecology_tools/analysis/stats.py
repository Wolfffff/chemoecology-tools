"""Statistical analysis functions for chemical ecology data."""

import numpy as np
import pandas as pd
from scipy import stats  # type: ignore

from ..core.gcms_experiment import GCMSExperiment


def calculate_enrichment_table(
    experiment: GCMSExperiment,
    group_column: str,
    class_column: str | None = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Calculate enrichment statistics for chemical compounds.

    Args:
        experiment: GCMSExperiment instance
        group_column: Column name for sample grouping
        class_column: Optional column name for chemical classification
        alpha: Significance level for statistical tests

    Returns:
        DataFrame with enrichment statistics

    Raises:
        ValueError: If group_column is not found in the data
    """
    # Validate inputs
    data = experiment.merge()
    if group_column not in data.columns:
        raise ValueError(f"Group column {group_column!r} not found in data")

    groups = data[group_column].unique()
    results = []

    for compound in experiment.chemical_cols:
        # Get chemical class if specified
        chemical_class = (
            experiment.get_chemical_property(compound, class_column)
            if class_column
            else None
        )

        # Rest of the function remains the same until the end
        group_stats = {}
        for group in groups:
            values = data[data[group_column] == group][compound]
            mean = values.mean()
            se = values.std() / np.sqrt(len(values))
            group_stats[group] = f"{mean:.2f} ± {se:.2f}"

        group_values = [data[data[group_column] == g][compound] for g in groups]
        _, p_val = stats.kruskal(*group_values)

        enriched_groups = []
        if p_val < alpha:
            for i, g1 in enumerate(groups):
                for g2 in groups[i + 1 :]:
                    _, p = stats.mannwhitneyu(
                        data[data[group_column] == g1][compound],
                        data[data[group_column] == g2][compound],
                        alternative="two-sided",
                    )
                    if p < alpha:
                        if (
                            data[data[group_column] == g1][compound].median()
                            > data[data[group_column] == g2][compound].median()
                        ):
                            enriched_groups.append(g1)
                        else:
                            enriched_groups.append(g2)

        group_bias = ", ".join(sorted(set(enriched_groups))) if enriched_groups else ""

        result = {
            "Chemical Class": chemical_class,
            "Compound": compound,
            "KW_pvalue": p_val,
            "Group Bias": group_bias,
            **group_stats,
        }
        results.append(result)

    df = pd.DataFrame(results)
    if class_column:
        df = df.sort_values(["Chemical Class", "Compound"])
    else:
        df = df.sort_values("Compound")

    return df
