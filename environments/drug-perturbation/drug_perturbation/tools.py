"""Compound identity and chemistry lookup, without biological reference answers."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
import verifiers.v1 as vf
from pydantic import Field
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, Lipinski, MolFromSmiles
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

DATA_DIR = Path(__file__).parent / "data"


@functools.lru_cache(maxsize=1)
def _load_compound_db() -> tuple[pd.DataFrame, list]:
    frame = pd.read_parquet(DATA_DIR / "compound_table.parquet")

    fingerprints: list = []
    for smiles in frame["smiles"]:
        molecule = MolFromSmiles(smiles)
        if molecule is None:
            fingerprints.append(None)
        else:
            fingerprints.append(
                AllChem.GetMorganFingerprintAsBitVect(
                    molecule,
                    radius=2,
                    nBits=2048,
                )
            )
    return frame, fingerprints


def _descriptors(molecule) -> dict[str, float]:
    return {
        "molecular_weight": round(Descriptors.MolWt(molecule), 1),
        "logp": round(Descriptors.MolLogP(molecule), 2),
        "tpsa": round(Descriptors.TPSA(molecule), 1),
        "num_h_bond_donors": Lipinski.NumHDonors(molecule),
        "num_h_bond_acceptors": Lipinski.NumHAcceptors(molecule),
        "num_rings": Lipinski.RingCount(molecule),
        "num_aromatic_rings": Lipinski.NumAromaticRings(molecule),
        "num_rotatable_bonds": Lipinski.NumRotatableBonds(molecule),
    }


def _murcko_scaffold(molecule) -> str:
    scaffold = MurckoScaffold.GetScaffoldForMol(molecule)
    return Chem.MolToSmiles(scaffold, canonical=True)


def identify_compound(smiles: str, top_k: int = 5) -> dict[str, Any]:
    """Look up structured metadata for a compound by its SMILES string.

    Args:
        smiles: SMILES of the query compound
        top_k: how many nearest neighbors to return (default 5)

    Returns:
        Dict with `exact_match`, `descriptors`, `scaffold`, `nearest_neighbors`.
        If SMILES is unparseable, returns {"error": "invalid SMILES"}.
    """
    if not 1 <= top_k <= 20:
        return {"error": "top_k must be between 1 and 20"}
    molecule = MolFromSmiles(smiles)
    if molecule is None:
        return {"error": "invalid SMILES"}

    frame, fingerprints = _load_compound_db()
    query_inchi_key = Chem.MolToInchiKey(molecule)

    exact_match = None
    if query_inchi_key:
        hit = frame[frame["inchi_key"] == query_inchi_key]
        if not hit.empty:
            exact_match = {"name": hit.iloc[0]["pert_iname"]}

    query_fingerprint = AllChem.GetMorganFingerprintAsBitVect(
        molecule,
        radius=2,
        nBits=2048,
    )
    similarities = []
    for index, fingerprint in enumerate(fingerprints):
        if fingerprint is None:
            continue
        similarity = DataStructs.TanimotoSimilarity(
            query_fingerprint,
            fingerprint,
        )
        similarities.append((similarity, index))
    similarities.sort(reverse=True)

    neighbors = []
    for similarity, index in similarities:
        if exact_match is not None and frame.iloc[index]["inchi_key"] == query_inchi_key:
            continue
        neighbors.append(
            {
                "name": frame.iloc[index]["pert_iname"],
                "similarity": round(similarity, 3),
            }
        )
        if len(neighbors) >= top_k:
            break

    return {
        "exact_match": exact_match,
        "descriptors": _descriptors(molecule),
        "scaffold": _murcko_scaffold(molecule),
        "nearest_neighbors": neighbors,
    }


class CompoundToolset(vf.Toolset[vf.ToolsetConfig]):
    """Expose the frozen compound lookup through Verifiers v1 MCP."""

    TOOL_PREFIX = None

    @vf.tool(name="identify_compound")
    def identify_compound(
        self,
        smiles: Annotated[str, Field(description="SMILES of the query compound")],
        top_k: Annotated[
            int,
            Field(description="number of nearest neighbors", ge=1, le=20),
        ] = 5,
    ) -> str:
        """Look up structured metadata for a compound by its SMILES string.

        Args:
            smiles: SMILES of the query compound
            top_k: how many nearest neighbors to return (default 5)

        Returns:
            Dict with `exact_match`, `descriptors`, `scaffold`, `nearest_neighbors`.
            If SMILES is unparseable, returns {"error": "invalid SMILES"}.
        """
        # Preserve the text representation used by the published benchmark.
        return repr(identify_compound(smiles=smiles, top_k=top_k))


if __name__ == "__main__":
    CompoundToolset.run()
