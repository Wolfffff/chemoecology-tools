"""Utility functions for fetching data from PubChem API."""

from functools import lru_cache
from typing import Any

import pubchempy as pcp  # type: ignore


@lru_cache(maxsize=1000)
def fetch_pubchem_data(chemical_name: str) -> dict[str, Any]:
    """Fetch and cache chemical data from PubChem API.

    Args:
        chemical_name: Name of chemical compound

    Returns:
        Dictionary of chemical properties
    """
    try:
        compounds = pcp.get_compounds(
            chemical_name, namespace="name", domain="compound"
        )
        if not compounds:
            return {}

        compound = compounds[0]
        output: dict[str, Any] = compound.to_dict()
        return output

    except Exception as e:
        print(f"Error fetching PubChem data for {chemical_name}: {e}")
        return {}
