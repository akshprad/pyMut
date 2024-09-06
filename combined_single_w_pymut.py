import argparse
import os
import sys
import re
import urllib.request
import warnings
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from Bio.PDB import PDBIO, PDBParser, Atom
from Bio.Align import PairwiseAligner
from uniprot import UniProt

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Define regular expression pattern for retrieving next link
re_next_link = re.compile(r'<(.+)>; rel="next"')

# Define retry settings for HTTP requests
retries = Retry(total=5, backoff_factor=0.25, status_forcelist=[500, 502, 503, 504])

# Create a session with retry settings
session = requests.Session()
session.mount("https://", HTTPAdapter(max_retries=retries))

# Function to retrieve the next link from HTTP response headers
def get_next_link(headers):
    if "Link" in headers:
        match = re_next_link.match(headers["Link"])
        if match:
            return match.group(1)

# Function to retrieve all isoforms for a given gene name
def get_all_isoforms(gene_name):
    isoforms = []
    url1 = "https://rest.uniprot.org/uniprotkb/search?query=reviewed:true+AND+"
    url3 = "&includeIsoform=true&format=list&(taxonomy_id:9606)"
    url = url1 + gene_name + url3
    batch_url = url
    while batch_url:
        response = session.get(batch_url)
        response.raise_for_status()
        for isoform in response.text.strip().split("\n"):
            isoform_url = f"https://www.uniprot.org/uniprot/{isoform}.fasta"
            isoform_response = session.get(isoform_url)
            isoform_response.raise_for_status()
            sequence = ''.join(isoform_response.text.strip().split('\n')[1:])
            isoforms.append((isoform, sequence))
        batch_url = get_next_link(response.headers)
    return isoforms

# Function to search for a specific residue at a position in isoforms of a gene
def search_residue(residue, position, gene_name):
    matching_isoforms = []
    all_isoforms = get_all_isoforms(gene_name)
    for isoform, sequence in all_isoforms:
        if len(sequence) > position - 1 and sequence[position - 1] == residue:
            matching_isoforms.append(isoform)
    return matching_isoforms

# Function to calculate the gene name for each isoform and filter isoforms with different gene names
def filter_isoforms_by_gene(matching_isoforms, gene_name):
    filtered_isoforms = []
    for isoform in matching_isoforms:
        gene_name_isoform = get_gene_name(isoform)
        if gene_name_isoform == gene_name:
            filtered_isoforms.append(isoform)
    return filtered_isoforms

def get_gene_name(uniprot_id):
    url = f"https://www.uniprot.org/uniprot/{uniprot_id}.txt"
    response = requests.get(url)
    lines = response.text.split("\n")
    for line in lines:
        if line.startswith("GN   Name="):
            gene_name = line.split("GN   Name=")[1].split(";")[0]
            return gene_name
    return None

def download_pdb(pdbcode, datadir, downloadurl="https://files.rcsb.org/download/"):
    pdbfn = pdbcode + ".pdb"
    url = downloadurl + pdbfn
    outfnm = os.path.join(datadir, pdbfn)
    try:
        urllib.request.urlretrieve(url, outfnm)
        return outfnm
    except Exception as err:
        print("ERROR:", err)
        return outfnm

def new_method_for_alphafold(pdbcode, datadir):
    new_url = "https://alphafold.ebi.ac.uk/files/AF-" + pdbcode + "-F1-model_v4.pdb"
    pdbfn2 = pdbcode + ".pdb"
    outfnm2 = os.path.join(datadir, pdbfn2)
    try:
        urllib.request.urlretrieve(new_url, outfnm2)
        return outfnm2
    except Exception as err:
        print("ERROR:", err)
        return outfnm2

# PyMUT functions
def read_sample_residue(residue_name):
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
                element=atom_name[0],
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

# Main script logic
if len(sys.argv) != 11:
    print("Usage: python file-name.py --gene-name gene-name --residue1 residue1 --position position --residue2 residue2 --top-isoforms True/False")
    sys.exit(1)

# Create argument parser
parser = argparse.ArgumentParser(description='Description of your program')

# Add arguments
parser.add_argument('--gene-name', type=str, help='Gene name')
parser.add_argument('--residue1', type=str, help='Residue 1')
parser.add_argument('--position', type=int, help='Position')
parser.add_argument('--residue2', type=str, help='Residue 2')
parser.add_argument('--top-isoforms', type=str, help='Show top isoforms: True/False')

# Parse the arguments
args = parser.parse_args()

# Retrieve command-line arguments
gene_name = args.gene_name
residue1 = args.residue1
position = args.position
residue2 = args.residue2
top_isoforms = args.top_isoforms

# Search for matching isoforms
matching_isoforms = search_residue(residue1, position, gene_name)

# Filter isoforms by gene name
matching_isoforms = filter_isoforms_by_gene(matching_isoforms, gene_name)

if len(matching_isoforms) > 0:
    selected_isoform = max(matching_isoforms, key=lambda isoform: len(isoform))
    isoform_id = selected_isoform[0:6]
    isoform_sequence = next((isoform[1] for isoform in matching_isoforms if isoform[0] == isoform_id), None)
    isoform_url = f"https://www.uniprot.org/uniprot/{isoform_id}"
else:
    print("Didn't find any matching isoforms for given selection.")
    sys.exit(0)

# Function to calculate similarity between two sequences using PairwiseAligner
def calculate_similarity(sequence1, sequence2):
    aligner = PairwiseAligner()
    aligner.mode = 'global'
    aligner.match_score = 1
    aligner.mismatch_score = -1
    alignments = aligner.align(sequence1, sequence2)
    best_alignment = alignments[0]
    alignment_score = best_alignment.score
    alignment_length = len(best_alignment)
    similarity = alignment_score / alignment_length * 100
    return similarity

# Function to score isoforms based on similarity to the gene sequence
def score_isoforms_by_similarity(gene_name, isoforms):
    scored_isoforms = []
    for isoform in isoforms:
        u = UniProt()
        entry = u.retrieve(isoform, "fasta")
        sequence = ''.join(entry.strip().split('\n')[1:])
        similarity = calculate_similarity(gene_name, sequence)
        scored_isoforms.append((isoform, similarity))
    scored_isoforms.sort(key=lambda x: x[1], reverse=True)
    return scored_isoforms[:3]  # Return the top 3 scored isoforms

# Retrieve the sequence for the selected isoforms and return the top 3 isoforms
top_scored_isoforms = score_isoforms_by_similarity(gene_name, matching_isoforms)

# Create UniProt object
u = UniProt()

# Retrieve the sequence for the selected isoform
sequence = u.retrieve(matching_isoforms[0:6], "fasta")
fasta_string = sequence
only_element = matching_isoforms[0]
uniprot_id = only_element[0:6]

# Define the URL for UniProt data
url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}"

# Make an HTTP GET request to the URL
response = requests.get(url)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    response_text = response.text

    # Function to extract PDB IDs
    def extract_pdb_ids(text):
        pattern = r'"database":"PDB","id":"([^"]+)"'
        matches = re.findall(pattern, text)
        return matches

    pdb_ids = extract_pdb_ids(response_text)
    print(pdb_ids[0])
else:
    print(f"Error: Failed to retrieve data. Status code: {response.status_code}")

# Download and mutate PDB file
current_dir = os.path.dirname(os.path.abspath(__file__))
pdbpath = download_pdb(pdb_ids[0], current_dir)
output_file = os.path.join(current_dir, 'mutated_' + pdb_ids[0] + '.pdb')

# Call mutate_residue function
mutate_residue(pdbpath, 'A', 140, residue2, output_file)
