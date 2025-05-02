print("Importing modules...")
import os, json
from argparse import ArgumentParser

from tagger.data.tools import make_data, load_data, to_ML
from tagger.train.train import save_test_data
from tagger.data.config import JET_FIELDS, INPUT_TAG, EXTRA_FIELDS, N_PARTICLES

parser = ArgumentParser()

parser.add_argument('-d', '--dir', default='./bkg_data', help= 'Where the background data is stored')
parser.add_argument('-i', '--input', default='bkg.root', help='The name of the input file to process')
parser.add_argument('-m', '--make_data', default=False, help='Make the data from root file or not')

args = parser.parse_args()
out_dir = args.dir

if args.make_data or (len(os.listdir(f"{out_dir}/chunks")) == 0):
    if (len(os.listdir(f"{out_dir}/chunks")) == 0): print("Chunks directory is empty, making data...")
    """
    If make data argument is true or the chunks directory is empty, then make the inputs from root file
    """
    print("Processing background data chunk by chunk...")
    make_data(
        infile=f"{out_dir}/{args.input}", 
        outdir=f"{out_dir}/chunks/",
        tag=INPUT_TAG,
        extras=EXTRA_FIELDS,
        jet_features_tag=JET_FIELDS,
        n_parts=N_PARTICLES,
        ratio=1.0,
        step_size="100MB",
        tree="outnano/Jets"
    )

print("Loading background data...")
_, data_test, class_labels, input_vars, extra_vars = load_data(f"{out_dir}/chunks/", percentage=100, test_ratio=1.0)    # Use all data as test data

with open(os.path.join(out_dir, "input_vars.json"), "w") as f: json.dump(input_vars, f, indent=4) #Dump output variables
with open(os.path.join(out_dir, "extra_vars.json"), "w") as f: json.dump(extra_vars, f, indent=4) #Dump output variables

print("Converting data to ML format...")
X_test, _, truth_pt_test, reco_pt_test, _, truth_mass_test, reco_mass_test = to_ML(data_test, class_labels)

print("Saving background data...")
save_test_data(out_dir, X_test, truth_pt_test, reco_pt_test, truth_mass_test, reco_mass_test, class_labels)

print("Background data processing complete!")