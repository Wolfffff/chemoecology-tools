"""Container for GCMS experimental data and metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ..utils.pubchem_utils import fetch_pubchem_data


class GCMSExperiment:
    """Gas Chromatography-Mass Spectrometry (GCMS) experimental data container.

    Manages GCMS abundance data, experimental metadata, and chemical properties.

    Attributes:
        abundance_df: DataFrame containing GCMS chemical abundance measurements
        metadata_df: DataFrame containing sample and experimental metadata
        id_col: Column name used to join abundance and metadata
        experiment_name: Optional identifier for the experiment
        chemical_metadata: Dictionary of chemical properties from config
    """

    def __init__(
        self,
        abundance_df: pd.DataFrame,
        metadata_df: pd.DataFrame,
        id_col: str = "ID",
        experiment_name: str | None = None,
        chemical_metadata: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """Initialize GCMS experiment container.

        Args:
            abundance_df: DataFrame with GCMS abundance measurements
            metadata_df: DataFrame with sample metadata
            id_col: Column name to join abundance and metadata
            experiment_name: Optional experiment identifier
            chemical_metadata: Optional dict mapping chemical names to properties

        """
        self.abundance_df = abundance_df
        self.metadata_df = metadata_df
        self.id_col = id_col
        self.experiment_name = experiment_name
        self.chemical_metadata = chemical_metadata or {}
        self._validate()
        self.chemical_cols = self._get_chemical_cols()

    @classmethod
    def from_files(
        cls,
        abundance_path: str | Path,
        metadata_path: str | Path,
        user_chemical_metadata: str | Path | None = None,
        fetch_pubchem: bool = True,
        id_col: str = "ID",
        filter_dict: dict[str, list[str]] | None = None,
        experiment_name: str | None = None,
    ) -> GCMSExperiment:
        """Create experiment from data files.

        Args:
            abundance_path: Path to abundance data file
            metadata_path: Path to metadata file
            user_chemical_metadata: Optional path to chemical properties YAML
            fetch_pubchem: Whether to fetch PubChem data for chemicals
            id_col: Column name to join on
            filter_dict: Optional filtering criteria {column: values_to_exclude}
            experiment_name: Optional experiment identifier

        Returns:
            New GCMSExperiment instance
        """
        abundance_df = pd.read_csv(abundance_path)
        metadata_df = pd.read_csv(metadata_path)
        chemical_cols = [col for col in abundance_df.columns if col != id_col]

        # Initialize chemical metadata
        chemical_metadata: dict[str, dict[str, Any]] = {}

        # Load user metadata if provided
        if user_chemical_metadata:
            with open(user_chemical_metadata, encoding="utf-8") as f:
                chemical_metadata = yaml.safe_load(f)

        # Fetch PubChem data if requested
        if fetch_pubchem:
            for chemical in chemical_cols:
                if chemical not in chemical_metadata:
                    chemical_metadata[chemical] = {}
                pubchem_data = fetch_pubchem_data(chemical)
                chemical_metadata[chemical].update(pubchem_data)

        if filter_dict:
            for col, values in filter_dict.items():
                if col in metadata_df.columns:
                    metadata_df = metadata_df[~metadata_df[col].isin(values)]
            abundance_df = abundance_df[abundance_df[id_col].isin(metadata_df[id_col])]

        return cls(
            abundance_df, metadata_df, id_col, experiment_name, chemical_metadata
        )

    def _validate(self) -> None:
        """Validate data consistency.

        Raises:
            ValueError: If validation fails
        """
        if self.id_col not in self.abundance_df.columns:
            raise ValueError(f"ID column {self.id_col!r} not found in abundance data")
        if self.id_col not in self.metadata_df.columns:
            raise ValueError(f"ID column {self.id_col!r} not found in metadata")

        # Validate chemical metadata if provided
        if self.chemical_metadata:
            unknown_chemicals = set(self._get_chemical_cols()) - set(
                self.chemical_metadata.keys()
            )
            if unknown_chemicals:
                print(f"WARNING: Unknown chemicals in metadata: {unknown_chemicals}")

    def _get_chemical_cols(self) -> list[str]:
        """Get chemical measurement columns from abundance data.

        Returns:
            List of column names excluding the ID column
        """
        return [col for col in self.abundance_df.columns if col != self.id_col]

    def merge(self) -> pd.DataFrame:
        """Merge abundance and metadata.

        Returns:
            DataFrame with joined abundance and metadata
        """
        return pd.merge(
            self.metadata_df, self.abundance_df, on=self.id_col, how="inner"
        )

    def filter_samples(self, criteria: dict[str, list[str]]) -> GCMSExperiment:
        """Filter samples based on metadata criteria.

        Args:
            criteria: Filtering criteria {column: values_to_exclude}

        Returns:
            New GCMSExperiment with filtered data
        """
        filtered_meta = self.metadata_df.copy()

        for col, values in criteria.items():
            if col in filtered_meta.columns:
                filtered_meta = filtered_meta[~filtered_meta[col].isin(values)]

        valid_ids = filtered_meta[self.id_col]
        filtered_abundance = self.abundance_df[
            self.abundance_df[self.id_col].isin(valid_ids)
        ]

        return GCMSExperiment(
            filtered_abundance,
            filtered_meta,
            self.id_col,
            self.experiment_name,
            self.chemical_metadata,
        )

    def get_abundance_matrix(self) -> pd.DataFrame:
        """Get chemical abundance matrix.

        Returns:
            DataFrame containing only chemical abundance measurements
        """
        return self.abundance_df[self.chemical_cols]

    def get_metadata(self, columns: list[str] | None = None) -> pd.DataFrame:
        """Get metadata columns.

        Args:
            columns: Optional list of column names to return

        Returns:
            DataFrame containing requested metadata columns
        """
        if columns is None:
            return self.metadata_df
        return self.metadata_df[columns]

    def get_chemical_property(
        self, chemical: str, property_name: str, default: Any = None
    ) -> Any:
        """Get property value for a chemical.

        Args:
            chemical: Name of the chemical
            property_name: Name of the property to retrieve
            default: Value to return if property not found

        Returns:
            Property value or default if not found
        """
        return self.chemical_metadata.get(chemical, {}).get(property_name, default)

    def get_chemicals_by_property(self, property_name: str, value: Any) -> list[str]:
        """Get chemicals that have a specific property value.

        Args:
            property_name: Name of the property to match
            value: Value to match

        Returns:
            List of chemical names with matching property
        """
        return [
            chem
            for chem, props in self.chemical_metadata.items()
            if props.get(property_name) == value
        ]

    def __len__(self) -> int:
        """Get number of samples.

        Returns:
            Integer count of samples in experiment
        """
        return len(self.abundance_df)

    def __str__(self) -> str:
        """Get string representation.

        Returns:
            String describing experiment contents
        """
        name = self.experiment_name or "Unnamed experiment"
        return (
            f"{name}: {len(self)} samples, "
            f"{len(self.chemical_cols)} chemicals measured"
        )

    def filter_trace_compounds(self, threshold: float = 0.005) -> GCMSExperiment:
        """Filter out trace chemical amounts below threshold.

        Args:
            threshold: Minimum abundance value to keep (lower values set to 0)

        Returns:
            GCMSExperiment with filtered abundance values

        Raises:
            ValueError: If threshold is not between 0 and 1
        """
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")

        filtered_abundance = self.abundance_df.copy()
        filtered_abundance[self.chemical_cols] = filtered_abundance[
            self.chemical_cols
        ].apply(lambda x: np.where(x < threshold, 0, x))

        return GCMSExperiment(
            filtered_abundance,
            self.metadata_df,
            self.id_col,
            self.experiment_name,
            self.chemical_metadata,
        )

    def calculate_relative_abundance(self) -> GCMSExperiment:
        """Calculate relative abundance of chemical compounds.

        Returns:
            GCMSExperiment with relative abundance values
        """
        relative_abundance = self.abundance_df.copy()
        row_sums = relative_abundance[self.chemical_cols].sum(axis=1)
        relative_abundance[self.chemical_cols] = relative_abundance[
            self.chemical_cols
        ].div(row_sums, axis=0)

        return GCMSExperiment(
            relative_abundance,
            self.metadata_df,
            self.id_col,
            self.experiment_name,
            self.chemical_metadata,
        )

    def filter(
        self,
        metadata_mask: pd.Series[bool] | None = None,
        chemical_mask: pd.Series[bool] | None = None,
    ) -> GCMSExperiment:
        """Filter experiment using boolean masks for metadata and/or chemicals.

        Args:
            metadata_mask: Boolean Series for filtering metadata rows
            chemical_mask: Boolean Series for filtering chemical columns

        Returns:
            New GCMSExperiment with filtered data

        Example:
            # Filter based on both metadata and chemicals
            meta_mask = exp.metadata_df["Species"].isin(["ant", "bee"])
            chem_mask = pd.Series([
                exp.get_chemical_property(c, "class") == "terpene"
                for c in exp.chemical_cols
            ], index=exp.chemical_cols)

            filtered_exp = exp.filter(
                metadata_mask=meta_mask,
                chemical_mask=chem_mask
            )
        """
        filtered_meta = self.metadata_df
        filtered_abundance = self.abundance_df

        # Apply metadata filtering
        if metadata_mask is not None:
            filtered_meta = filtered_meta[metadata_mask]
            filtered_abundance = filtered_abundance[
                filtered_abundance[self.id_col].isin(filtered_meta[self.id_col])
            ]

        # Apply chemical filtering
        if chemical_mask is not None:
            selected_chemicals = [
                col
                for col, include in chemical_mask.items()
                if include and col in self.chemical_cols
            ]
            filtered_abundance = filtered_abundance[[self.id_col] + selected_chemicals]

        return GCMSExperiment(
            filtered_abundance,
            filtered_meta,
            self.id_col,
            self.experiment_name,
            self.chemical_metadata,
        )
