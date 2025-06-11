# Python
import os, gc, json, glob, shutil

# Third party
import numpy as np
import awkward as ak
import tensorflow as tf
import uproot, yaml

# Dataset configuration
from .config import FILTER_PATTERN, N_PARTICLES, INPUT_TAG, EXTRA_FIELDS, JET_FIELDS

gc.set_threshold(0)

#>>>>>>>>>>>>>>>>>>>PRIVATE FUNCTIONS<<<<<<<<<<<<<<<<<<<<<<
def _add_response_vars(data):
    data['jet_ptUncorr_div_ptGen'] = ak.nan_to_num(data['jet_pt_phys']/data['jet_genmatch_pt'], copy=True, nan=0.0, posinf=0., neginf=0.)
    data['jet_ptCorr_div_ptGen'] = ak.nan_to_num(data['jet_pt_corr']/data['jet_genmatch_pt'], copy=True, nan=0.0, posinf=0., neginf=0.)
    data['jet_ptRaw_div_ptGen'] = ak.nan_to_num(data['jet_pt_raw']/data['jet_genmatch_pt'], copy=True, nan=0.0, posinf=0., neginf=0.)

def _define_target(data):
    """
    Splits data by particle flavor and applies conditions for each category. Also creates the pT target.

    Parameters:
        data (awkward array): The input data to split.

    Returns:
        dict: A dictionary containing the split data by label.
    """

    genmatch_base = (data['jet_genmatch_pt'] > 0) | (data['jet_genmatch_mass'] > 0)    # Only jets matched to a gen jet
    data = data[genmatch_base]

    # clipped_l1_mass, clipped_gen_mass = np.clip( data["jet_mass"], 1, 128 ), np.clip( data["jet_genmatch_mass"], 1, 128 )
    # log_l1_mass, log_gen_mass = np.log2( clipped_l1_mass ), np.log2( clipped_gen_mass )
    # clipped_log_mass_ratio = np.clip( log_gen_mass / log_l1_mass, 0.5, 2 )

    pt_ratio = ak.nan_to_num( data["jet_genmatch_pt"] / data["jet_pt_phys"], nan=0, posinf=0, neginf=0)
    data['target_pt'] = np.clip(pt_ratio, 0.3, 3)
    data['target_pt_phys'] = np.clip( ak.nan_to_num( data["jet_genmatch_pt"], nan=0, posinf=0, neginf=0 ), 0, 2000)

    mass_ratio = ak.nan_to_num( data["jet_genmatch_mass"] / data["jet_mass"], nan=0, posinf=0, neginf=0)
    data["target_mass"] = np.clip(mass_ratio, 0.3, 3)
    data['target_mass_phys'] = np.clip( ak.nan_to_num( data["jet_genmatch_mass"], nan=0, posinf=0, neginf=0 ), 0, 182)

    # Apply pt_cut and mass_cut
    jet_ptmin_gen, jet_massmin_gen = (data['target_pt_phys'] > 15.0), (data['target_mass_phys'] > 5.0)

    return data[jet_ptmin_gen & jet_massmin_gen]
    

def _get_pfcand_fields(tag):
    
    # Get the directory of the current file (tools.py)
    current_dir = os.path.dirname(__file__)

    # Construct the path to pfcand_fields.yml relative to tools.py
    pfcand_fields_path = os.path.join(current_dir, "pfcand_fields.yml")

    # Load the YAML file as a dictionary
    with open(pfcand_fields_path, "r") as file: pfcand_fields = yaml.safe_load(file)

    return pfcand_fields[tag]

def _pad_fill(array, target):
    '''
    pad an array to target length and then fill it with 0s
    '''
    return ak.fill_none(ak.pad_none(array, target, axis=1, clip=True), 0)

def _make_nn_inputs(data_split, tag, n_parts):

    features = _get_pfcand_fields(tag)

    #Concatenate all the inputs
    inputs_list = []

    #Vertically stacked them to create input sets
    #https://awkward-array.org/doc/main/user-guide/how-to-restructure-concatenate.html
    #Also pad and fill them with 0 to the number of constituents we are using (nconstit)
    for field in features:
        field_array = data_split["jet_pfcand"][field]

        padded_filled_array = _pad_fill(field_array, n_parts)
        inputs_list.append(padded_filled_array[:, :, np.newaxis])

    from math import pi
    pt = data_split["jet_pfcand"]["pt"]
    deta = data_split["jet_pfcand"]["deta"]
    dphi = data_split["jet_pfcand"]["dphi"]
    
    energy = pt * np.cosh(deta*pi/720)
    px = pt * np.cos(dphi*pi/720)
    py = pt * np.sin(dphi*pi/720)
    pz = pt * np.sinh(deta*pi/720)

    inputs_list.append(_pad_fill(energy, n_parts)[:, :, np.newaxis])
    inputs_list.append(_pad_fill(px, n_parts)[:, :, np.newaxis])
    inputs_list.append(_pad_fill(py, n_parts)[:, :, np.newaxis])
    inputs_list.append(_pad_fill(pz, n_parts)[:, :, np.newaxis])

    #batch_size, n_particles, n_features
    inputs = ak.concatenate(inputs_list, axis=2)
    data_split['nn_inputs'] = inputs
    
    return

def _make_nn_jet_inputs(data_split, tag):
    
    features = _get_pfcand_fields(tag)

    #Concatenate all the inputs
    inputs_list = []

    #Vertically stacked them to create input sets
    #https://awkward-array.org/doc/main/user-guide/how-to-restructure-concatenate.html
    #Also pad and fill them with 0 to the number of constituents we are using (nconstit)
    for field in features:
        field_array = data_split[field]
        inputs_list.append(field_array[:, np.newaxis])

    #batch_size, n_particles, n_features
    inputs = ak.concatenate(inputs_list, axis=1)
    data_split['nn_jet_inputs'] = inputs
    
    return

def _save_chunk_metadata(metadata_file, chunk, entries, outfile):

    chunk_info = {
        "chunk": chunk,
        "entries": entries,
        "file": outfile
    }

    # Load existing metadata or start a new list
    if os.path.exists(metadata_file):
        with open(metadata_file, "r") as f:
            content = f.read()
            metadata = json.loads(content) if content.strip() else []
    else:
        metadata = []

    metadata.append(chunk_info)

    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=4)

    return

def _save_dataset_metadata(outdir, tag, extras, jet_features_tag):

    dataset_metadata_file = os.path.join(outdir, 'variables.json')

    if jet_features_tag is None:
        metadata = {"outputs": ["pt", "mass"],
                "inputs": _get_pfcand_fields(tag),
                "extras": _get_pfcand_fields(extras)}
    else:
        metadata = {"outputs": ["pt", "mass"],
                    "inputs": _get_pfcand_fields(tag),
                    "extras": _get_pfcand_fields(extras),
                    "jet_level_inputs": _get_pfcand_fields(jet_features_tag)}

    with open(dataset_metadata_file, "w") as f: json.dump(metadata, f, indent=4)

    return

def _process_chunk(data_split, tag, extras, jet_features_tag, n_parts, chunk, outdir):
    """
    Process chunk of data_split to save/parse it for training datasets
    """

    #Create the NN inputs
    _make_nn_inputs(data_split, tag, n_parts)
    extra_features = _get_pfcand_fields(extras)

    #Save them to a root file
    # save_fields=['nn_inputs', 'class_label', 'target_pt', 'target_pt_phys', 'target_mass', 'target_mass_phys'] + extra_features
    save_fields=['nn_inputs', 'target_pt', 'target_pt_phys', 'target_mass', 'target_mass_phys'] + extra_features    #target_pt_phys

    if jet_features_tag is not None:
        _make_nn_jet_inputs(data_split, jet_features_tag)
        save_fields += ["nn_jet_inputs"]

    # Filter the data_split to only include save_fields
    filtered_data = {field: data_split[field] for field in save_fields}

    #Save chunk to files
    outfile = os.path.join(outdir, f'data_chunk_{chunk}.root')
    with uproot.recreate(outfile) as f:
        f["data"] = filtered_data
        print(f"Saved chunk {chunk} to {outfile}")

    # Log metadata
    metadata_file = os.path.join(outdir, "metadata.json")
    _save_chunk_metadata(metadata_file, chunk, len(data_split), outfile) #Chunk, Entries, Outfile

    #Delete the variables to save memory
    gc.collect()

    return

# >>>>>>FUNCTIONS THAT SHOULD BE USED EXTERNALLY!<<<<<<<
def extract_array(tree, field, entry_stop):
    """
    Extracts an array from the tree with a limit on the number of entries.
    """
    return tree[field].array(entry_stop=entry_stop)

def extract_nn_inputs(data, input_vars, n_parts=16 ,n_entries=None):
    """
    Extract nn inputs based on the input_vars list
    """

    #Concatenate all the inputs
    inputs_list = []

    for field in input_vars:

        field_array = extract_array(data, f"jet_pfcand_{field}", n_entries)

        padded_filled_array = _pad_fill(field_array, n_parts)
        inputs_list.append(padded_filled_array[:,:,np.newaxis])

    #batch_size, n_particles, n_features
    inputs = ak.concatenate(inputs_list, axis=2)

    return inputs

def group_id_values(event_id, *arrays, num_elements = 2):
    '''
    Group values according to event id.
    Filter out events that has less than num_elements
    '''

    # Use ak.argsort to sort based on event_id
    sorted_indices = ak.argsort(event_id)
    sorted_event_id = event_id[sorted_indices]

    # Find unique event_ids and counts manually
    unique_event_id, counts = np.unique(sorted_event_id, return_counts=True)
    
    # Use ak.unflatten to group the arrays by counts
    grouped_id = ak.unflatten(sorted_event_id, counts)
    grouped_arrays = [ak.unflatten(arr[sorted_indices], counts) for arr in arrays]

    #Filter out groups that don't have at least num_elements elements
    mask = ak.num(grouped_id) >= num_elements
    filtered_grouped_arrays = [arr[mask] for arr in grouped_arrays]

    return grouped_id[mask], filtered_grouped_arrays

def to_ML(data, use_jets):
    """
    Take in the data from make_data (loaded by load_data) and make them ready for training.
    """
    keepExtras = False
    constit_feats = np.asarray(data["nn_inputs"]) if keepExtras else np.asarray(data["nn_inputs"])[:,:-4]    # exclude E, px, py and pz

    if use_jets:
        try:
            features = ( constit_feats, np.asarray(data['nn_jet_inputs']) )
            # features = data["nn_inputs"], data["nn_jet_inputs"]
        except KeyError: raise KeyError("Error: jet-level features not found in data. Please check your dataset or the tag used.")
    else:
        features = constit_feats
        # features = data["nn_inputs"]
    
    pt_target = np.asarray(data['target_pt'])
    truth_pt = np.asarray(data['target_pt_phys'])
    reco_pt = np.asarray(data['jet_pt_phys'])

    mass_target = np.asarray(data['target_mass'])
    truth_mass = np.asarray(data['target_mass_phys'])
    reco_mass = np.asarray(data['jet_mass'])

    return features, pt_target, truth_pt, reco_pt, mass_target, truth_mass, reco_mass

def load_data(outdir, percentage, test_ratio=0.15, fields=None):
    """
    Load a specified percentage of the dataset using uproot.concatenate.

    Parameters:
        outdir (str): The output directory containing the data chunks.
        percentage (float): The percentage of TOTAL data to load (0-100).
        test_ratio (float): how much of the total data would be used for testing (0-1)
        fields (list, optional): Specific fields to load. If None, load all fields.

    Returns:
        awkward.Array: Concatenated data arrays from selected chunks.
    """
    import json
    import numpy as np

    print("Loading data from: ", outdir)

    # Load metadata to determine chunks to load
    metadata_file = os.path.join(outdir, "metadata.json")
    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    total_chunks = len(metadata)
    print(total_chunks, " chunks found in metadata")
    chunks_to_load = int(np.floor((percentage / 100) * total_chunks))
    print(f"Loading {chunks_to_load} chunks out of {total_chunks} ({percentage}%)")

    # Use uproot.concatenate to load and combine data from multiple files
    # data = uproot.concatenate(chunk_files, filter_name=fields, library="ak")
    chunk_files = [metadata[int(i*np.floor((1 / (percentage/100))))]["file"] + ":data" for i in range(chunks_to_load)]
    data = uproot.concatenate(chunk_files, filter_name=fields, library="ak")

    print(data)
    # Shuffle the data indices
    total_data_len = len(data)
    indices = np.arange(total_data_len)
    np.random.shuffle(indices)

    # Split indices based on test_ratio
    split_index = int((1 - test_ratio) * total_data_len)
    train_indices, test_indices = indices[:split_index], indices[split_index:]

    # Split the data into training and testing sets
    train_data = data[train_indices]
    test_data = data[test_indices]

    #Load corresponding metadata for classlabels/input variables
    data_metadata_file  = os.path.join(outdir, "variables.json")
    with open(data_metadata_file, "r") as f:
        variables = json.load(f)
        targets = variables['outputs']
        input_vars = variables['inputs']
        extra_vars = variables['extras']
    
    return train_data, test_data, targets, input_vars, extra_vars

def make_data(infile='/eos/home-l/lroberts/mass_regression/CMSSW_14_2_0_pre2/src/TrainTagger/sc8Jets.root', 
              outdir='training_data/',
              tag=INPUT_TAG,
              extras=EXTRA_FIELDS,
              jet_features_tag=JET_FIELDS,
              n_parts=N_PARTICLES,
              ratio=1.0,
              step_size="100MB",
              tree="outnano/Jets"):
    """
    Process the data set in chunks from the input ntuples file.

    Parameters:
        infile (str): The input file path.
        outdir (str): The output directory.
        tag (str): Input tags to use from pfcands, defined in pfcand_fields.yml.
        extras (str): Extra fields to store for plotting, defined in pfcand_fields.yml
        jet_features_tag (str): Jet features to use from pfcands, defined in pfcand_fields.yml.
        n_parts (int): Number of constituent particles to use for tagging.
        fraction (float) : fraction from (0-1) of data to process for training/testing
        step_size (str): Step size for uproot iteration.
    """

    #Check if output dir already exists, remove if so
    if os.path.exists(outdir):
        confirm = input(f"The directory '{outdir}' already exists. Do you want to delete it and continue? [y/n]: ")
        if confirm.lower() == 'y':
            shutil.rmtree(outdir)
            print(f"Deleted existing directory: {outdir}")
        else:
            print("Exiting without making changes.")
            return

    #Create output training dataset
    os.makedirs(outdir, exist_ok=True)
    print("Output directory:", outdir)

    #Loop through the entries
    num_entries = uproot.open(infile)[tree].num_entries
    num_entries_done = 0
    removedByL1Cuts = 0
    removedByGenSel = 0
    chunk = 0
    print(f"Total entries: {num_entries}")
    for data in uproot.iterate(infile, filter_name=FILTER_PATTERN, how="zip", step_size=step_size, max_workers=8):
        
        nEntries = len(data)
        #Define basic jet kinematic cuts
        jet_cut = (data['jet_pt_phys'] > 15) & (np.abs(data['jet_eta_phys']) < 2.4) & (data['jet_reject'] == 0) & (data['jet_mass'] > 5)
        data = data[jet_cut]
        nEntriesAfterL1Cuts = len(data)
        removedByL1Cuts += (nEntries - nEntriesAfterL1Cuts)    # num_entries_survived += len(data)

        data = _define_target(data)
        nEntriesAfterL1AndGenCuts = len(data)
        removedByGenSel += (nEntriesAfterL1Cuts - nEntriesAfterL1AndGenCuts)

        if nEntriesAfterL1AndGenCuts == 0:
            print(f"Warning! No entries survived the cuts and selection in chunk {chunk}.")
            print(f"    Initial number of entries in chunk: {nEntries}")
            print(f"    Entries removed by level-1 cuts: {removedByL1Cuts}")
            print(f"    Entries removed by gen selection: {removedByGenSel}")
            print(f"    Total entries remaining in chunk: {nEntriesAfterL1AndGenCuts}")

        #If first chunk then save metadata of the dataset
        if chunk == 0: _save_dataset_metadata(outdir, tag, extras, jet_features_tag)

        #Process and save training data for a given feature set
        _process_chunk(data, tag=tag, extras=extras, jet_features_tag=jet_features_tag, n_parts=n_parts, chunk=chunk, outdir=outdir)

        #Number of chunk for indexing files
        chunk += 1
        num_entries_done += nEntries
        print(f"Processed {num_entries_done}/{num_entries} entries | {np.round(num_entries_done / num_entries * 100, 1)}%")
        if num_entries_done / num_entries >= ratio: break

    print(f"\nTotal entries processed: {num_entries_done} out of {num_entries}")
    print(f"{removedByL1Cuts} entries removed by level-1 cuts")
    print(f"{removedByGenSel} entries removed by subsequent gen selection")
    print(f"Total entries saved: {num_entries_done - removedByL1Cuts - removedByGenSel} out of {num_entries_done}")
