"""Core analysis functions for dimensionality reduction and pattern detection."""

from typing import Any

import pandas as pd
import skbio.stats.composition as composition  # type: ignore
from sklearn.decomposition import PCA  # type: ignore
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # type: ignore
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.manifold import MDS  # type: ignore
from sklearn.metrics import classification_report  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore

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
        nmds_coords, columns=[f"NMDS{i + 1}" for i in range(n_components)]
    )


def calculate_compositional_stats(
    experiment: GCMSExperiment, chemical_cols: list[str] | None = None
) -> dict[str, Any]:
    """Calculate compositional statistics including CLR transform.

    Args:
        experiment: GCMSExperiment instance
        chemical_cols: Optional list of chemical columns to analyze

    Returns:
        dict containing CLR transformed data and other stats
    """
    # Get abundance data
    abund_data = experiment.abundance_df.copy()

    if chemical_cols is None:
        chemical_cols = experiment.chemical_cols

    abund_data = abund_data[chemical_cols]

    # CLR transform
    clr_transformed = pd.DataFrame(
        composition.clr(abund_data.replace(0, 1e-6).values),
        columns=abund_data.columns,
        index=abund_data.index,
    )

    # Scale data
    scaler = StandardScaler()
    clr_scaled = scaler.fit_transform(clr_transformed)

    return {
        "raw_data": abund_data,
        "clr_transformed": clr_transformed,
        "clr_scaled": clr_scaled,
        "scaler": scaler,
    }


def perform_pca(transformed_data: dict[str, Any]) -> dict[str, Any]:
    """Perform PCA on transformed chemical data.

    Args:
        transformed_data: Output from calculate_compositional_stats()

    Returns:
        dict containing PCA results
    """
    pca = PCA()
    pca_result = pca.fit_transform(transformed_data["clr_scaled"])

    return {
        "coords": pd.DataFrame(
            pca_result[:, :2],
            columns=["PC1", "PC2"],
        ),
        "explained_variance": pca.explained_variance_ratio_,
        "pca_obj": pca,
    }


def perform_lda(
    transformed_data: dict[str, Any],
    experiment: GCMSExperiment,
    group_col: str = "Caste",
    test_size: float = 0.3,
    n_components: int = None
) -> dict[str, Any]:
    """Perform LDA on transformed chemical data.

    Args:
        transformed_data: Output from calculate_compositional_stats()
        experiment: GCMSExperiment instance
        group_col: Column name for grouping variable
        test_size: Proportion of data to use for testing

    Returns:
        dict containing LDA results
    """
    x = transformed_data["clr_scaled"]
    y = experiment.metadata_df[group_col]

    # Split data
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=42
    )

    # Fit LDA
    lda = LinearDiscriminantAnalysis(n_components = n_components)
    lda.fit(x_train, y_train)
    y_pred = lda.predict(x_test)

    # Transform full dataset
    lda_result = lda.transform(x)

    return {
        "coords": pd.DataFrame(
            lda_result,
            columns=[f"LD{i + 1}" for i in range(lda_result.shape[1])],
            index=experiment.metadata_df.index,
        ),
        "classification_report": classification_report(y_test, y_pred),
        "lda_obj": lda,
        "training_data": (x_train, x_test, y_train, y_test),
    }


def perform_random_forest(
    transformed_data: dict[str, Any],
    experiment: GCMSExperiment,
    group_col: str = "Caste",
    test_size: float = 0.3,
) -> dict[str, Any]:
    """Perform Random Forest classification on chemical data.

    Args:
        transformed_data: Output from calculate_compositional_stats()
        experiment: GCMSExperiment instance
        group_col: Column name for grouping variable
        test_size: Proportion of data to use for testing

    Returns:
        dict containing RF results
    """
    x = transformed_data["clr_scaled"]
    y = experiment.metadata_df[group_col]

    # Split data
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=42
    )

    # Fit RF
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_test)

    feature_cols = transformed_data["clr_transformed"].columns

    return {
        "feature_importance": pd.DataFrame(
            {"feature": feature_cols, "importance": rf.feature_importances_}
        ).sort_values("importance", ascending=False),
        "classification_report": classification_report(y_test, y_pred),
        "rf_obj": rf,
        "training_data": (x_train, x_test, y_train, y_test),
    }
