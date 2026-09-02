import io

from rdkit import Chem
from rdkit.Chem import rdDetermineBonds


def atoms_to_rdkit_mol(struct, charge=0):
    """
    Converts an ASE structure to an RDKit Mol object.
    Uses RDKit's XYZ parser and determines covalent bonds via 3D coordinates.

    :param struct: ASE Atoms structure object.
    :type struct: ase.Atoms
    :param charge: Total formal charge of the molecule.
    :type charge: int
    :return: RDKit Mol object containing 3D coordinates and perceived bonds.
    :rtype: rdkit.Chem.Mol
    """
    with io.StringIO() as buffer:
        from ase.io import write

        write(buffer, struct, format="xyz")
        xyz_block = buffer.getvalue()

    mol = Chem.MolFromXYZBlock(xyz_block)
    if mol is None:
        raise ValueError("RDKit failed to parse XYZ coordinates from struct.")

    rdDetermineBonds.DetermineBonds(mol, charge=charge)
    return mol


def get_canonical_smiles(ase_st=None, rdmol=None, charge=0):
    """
    Returns a canonical SMILES string from an ASE structure or an RDKit Mol.
    Exactly one of `ase_st` or `rdmol` must be provided.

    :param ase_st: Optional ASE structure.
    :type ase_st: ase.Atoms
    :param rdmol: Optional RDKit Mol object.
    :type rdmol: rdkit.Chem.Mol
    :param charge: Total formal charge (only used if parsing ase_st).
    :type charge: int
    :return: Canonical SMILES string.
    :rtype: str
    """
    if (ase_st is None) == (rdmol is None):
        raise ValueError("Exactly one of 'ase_st' or 'rdmol' must be given.")

    if ase_st is not None:
        mol = atoms_to_rdkit_mol(ase_st, charge=charge)
    else:
        mol = rdmol

    return Chem.MolToSmiles(mol)


def write_sdf(mol_list, output_path, properties_list=None):
    """
    Writes a collection of RDKit Mol objects to an SDF file with properties.

    :param mol_list: Sequence of RDKit Mol objects with 3D coordinates.
    :type mol_list: list[rdkit.Chem.Mol]
    :param output_path: Path to target destination .sdf file.
    :type output_path: str or pathlib.Path
    :param properties_list: Optional list of property dictionaries per mol.
    :type properties_list: list[dict]
    """
    writer = Chem.SDWriter(str(output_path))
    for i, mol in enumerate(mol_list):
        if properties_list and i < len(properties_list):
            props = properties_list[i]
            for key, val in props.items():
                mol.SetProp(str(key), str(val))
        writer.write(mol)
    writer.close()
