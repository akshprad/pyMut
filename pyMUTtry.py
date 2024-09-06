from Bio.PDB import PDBIO, PDBParser, Atom
import numpy as np
import os
import logging

logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s', datefmt='%H:%M')

DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')

# Define your residue information as provided
VW_RADII = { ... }  # as defined in your code
CHI_ANGLES = { ... }  # as defined in your code
RESIDUE_ORDER = { ... }  # as defined in your code

def read_sample_residue(residue_name):
    """
    Prepare a sample residue with dummy coordinates for mutation purposes.
    """
    # Define dummy coordinates for TYR residue
    dummy_residues = {
        'TYR': {
            'N': [0.0, 0.0, 0.0],
            'CA': [1.5, 0.0, 0.0],
            'C': [2.5, 1.5, 0.0],
            'O': [3.0, 2.0, 0.0],
            'CB': [1.5, -1.5, 0.0],
            'CG': [2.5, -2.5, 0.0],
            'CD1': [3.5, -2.5, 1.0],
            'CD2': [3.5, -2.5, -1.0],
            'CE1': [4.5, -3.0, 1.0],
            'CE2': [4.5, -3.0, -1.0],
            'CZ': [5.5, -3.0, 0.0],
            'OH': [6.0, -3.5, 0.0]
        }
    }
    sample_residue = dummy_residues.get(residue_name, {})
    return sample_residue

def is_backbone(atom):
    """
    Check if the given atom is part of the protein backbone.
    
    :param atom: A Biopython Atom object.
    :return: True if the atom is part of the backbone (N, CA, C, O), False otherwise.
    """
    return atom.get_id() in ['N', 'CA', 'C', 'O']

def mutate_residue(pdb_file, chain_id, residue_id, new_residue_name, output_file, rotamer_lib=None, mutation_type="best"):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('structure', pdb_file)
    model = structure[0]
    chain = model[chain_id]
    
    try:
        residue = chain[residue_id]
    except KeyError:
        raise KeyError(f"Residue {residue_id} not found in chain {chain_id}!")
    
    # Remove old residue atoms
    for atom in list(residue.get_atoms()):
        if not is_backbone(atom):
            atom.get_parent().detach_child(atom.get_id())
    
    # Get and prepare the sample residue
    sample_residue = read_sample_residue(new_residue_name)
    
    # Add new atoms to the residue
    for atom_name, coord in sample_residue.items():
        if atom_name not in ['C', 'N', 'CA', 'O']:
            new_atom = Atom.Atom(
                name=atom_name,
                coord=np.asarray(coord),
                fullname="{}{}".format(" " * (4 - len(atom_name)), atom_name),
                bfactor=1.0,
                altloc=" ",
                occupancy=1.0,
                serial_number=9999
            )
            residue.add(new_atom)
    
    residue.resname = new_residue_name
    
    # Save the new structure to a file
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_file)
    logging.info(f"Mutation complete. New PDB file saved as {output_file}")

# Example usage
pdb_file = '3biw.pdb'
chain_id = 'A'
residue_id = 140
new_residue_name = 'TYR'
output_file = '3biw_mutated.pdb'

mutate_residue(pdb_file, chain_id, residue_id, new_residue_name, output_file)
