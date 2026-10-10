from __future__ import annotations

from typing import Any


ASPIRIN_SMILES = "CC(=O)Oc1ccccc1C(=O)O"


def calculate_descriptors(smiles: str) -> dict[str, Any]:
    """Calculate a small descriptor panel for a SMILES string if RDKit exists."""
    try:
        from rdkit import Chem  # type: ignore
        from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors  # type: ignore
    except Exception as exc:
        return {
            "ok": False,
            "tool": "rdkit_descriptors",
            "available": False,
            "input": {"smiles": smiles},
            "error": f"RDKit unavailable: {exc}",
        }

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            "ok": False,
            "tool": "rdkit_descriptors",
            "available": True,
            "input": {"smiles": smiles},
            "valid_smiles": False,
            "error": "Invalid SMILES",
        }

    descriptors = {
        "canonical_smiles": Chem.MolToSmiles(mol, canonical=True),
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "molecular_weight": Descriptors.MolWt(mol),
        "exact_molecular_weight": Descriptors.ExactMolWt(mol),
        "logp": Crippen.MolLogP(mol),
        "tpsa": rdMolDescriptors.CalcTPSA(mol),
        "hbond_donors": Lipinski.NumHDonors(mol),
        "hbond_acceptors": Lipinski.NumHAcceptors(mol),
        "rotatable_bonds": Lipinski.NumRotatableBonds(mol),
        "heavy_atom_count": mol.GetNumHeavyAtoms(),
        "ring_count": rdMolDescriptors.CalcNumRings(mol),
    }
    return {
        "ok": True,
        "tool": "rdkit_descriptors",
        "available": True,
        "input": {"smiles": smiles},
        "valid_smiles": True,
        "descriptors": descriptors,
    }


def smoke_test() -> dict[str, Any]:
    return calculate_descriptors(ASPIRIN_SMILES)
