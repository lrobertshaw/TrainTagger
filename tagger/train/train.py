from argparse import ArgumentParser
import os, shutil, json

#Import from other modules
from tagger.data.tools import make_data, load_data, to_ML
from tagger.plot.basic import loss_history, basic
import tagger.train.models as models

#Third parties
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.utils import plot_model
from sklearn.utils.class_weight import compute_class_weight
import mlflow

from datetime import datetime
import inspect

num_threads = 24
os.environ["OMP_NUM_THREADS"] = str(num_threads)
os.environ["TF_NUM_INTRAOP_THREADS"] = str(num_threads)
os.environ["TF_NUM_INTEROP_THREADS"] = str(num_threads)

tf.config.threading.set_inter_op_parallelism_threads(num_threads)
tf.config.threading.set_intra_op_parallelism_threads(num_threads)

# GLOBAL PARAMETERS TO BE DEFINED WHEN TRAINING
tf.keras.utils.set_random_seed(420) #not a special number 
BATCH_SIZE = 256 #1024
EPOCHS = 200
VALIDATION_SPLIT = 0.2

# Sparsity parameters
I_SPARSITY = 0.0 #Initial sparsity
F_SPARSITY = 0.1 #Final sparsity

def prune_model(model, num_samples):
    """
    Pruning settings for the model. Return the pruned model
    """

    print("Begin pruning the model...")

    #Calculate the ending step for pruning
    end_step = np.ceil(num_samples / BATCH_SIZE).astype(np.int32) * EPOCHS

    #Define the pruned model
    # pruning_params = {'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=I_SPARSITY, final_sparsity=F_SPARSITY, begin_step=0, end_step=end_step)}
    pruned_model = model#tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)

    mets = ['categorical_crossentropy', 'accuracy']
    pruned_model.compile(optimizer='adam',
                         loss = {'jet_id_output': 'categorical_crossentropy'},
                         metrics = {'jet_id_output': mets},
                         weighted_metrics = {'jet_id_output': mets}
                         )

    return pruned_model

def save_test_data(out_dir, X_test, y_test, truth_pt_test, reco_pt_test, truth_mass_test, reco_mass_test, class_labels, use_jets):

    os.makedirs(os.path.join(out_dir,'testing_data'), exist_ok=True)

    if use_jets:
        np.save(os.path.join(out_dir, "testing_data/X_test_constits.npy"), X_test[0])
        np.save(os.path.join(out_dir, "testing_data/X_test_jets.npy"), X_test[1])
    else:
        np.save(os.path.join(out_dir, "testing_data/X_test_constits.npy"), X_test)

    np.save(os.path.join(out_dir, "testing_data/y_test.npy"), y_test)

    np.save(os.path.join(out_dir, "testing_data/truth_pt_test.npy"), truth_pt_test)
    np.save(os.path.join(out_dir, "testing_data/reco_pt_test.npy"), reco_pt_test)

    np.save(os.path.join(out_dir, "testing_data/truth_mass_test.npy"), truth_mass_test)
    np.save(os.path.join(out_dir, "testing_data/reco_mass_test.npy"), reco_mass_test)

    with open(os.path.join(out_dir, "class_label.json"), "w") as f: json.dump(class_labels, f, indent=4) #Dump output variables

    print(f"Test data saved to {out_dir}")


def train(out_dir, percent, model_name, use_jets):

    #Remove output dir if exists
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
        print(f"Re-created existing directory: {out_dir}.")

    #Create dir to save results
    os.makedirs(out_dir)

    #Load the data, class_labels and input variables name, not really using input variable names to be honest
    data_train, data_test, class_labels, input_vars, extra_vars = load_data("training_data/", percentage=percent)
    print(f"Loaded {len(data_train)} training jets and {len(data_test)} testing jets")
    
    #Save input variables and extra variables metadata
    with open(os.path.join(out_dir, "input_vars.json"), "w") as f: json.dump(input_vars, f, indent=4) #Dump output variables
    with open(os.path.join(out_dir, "extra_vars.json"), "w") as f: json.dump(extra_vars, f, indent=4) #Dump output variables

    #Make into ML-like data for training
    X_train, y_train, pt_target_train, truth_pt_train, reco_pt_train, mass_target_train, truth_mass_train, reco_mass_train = to_ML(data_train, class_labels, use_jets)
    
    #Save X_test, y_test, and truth_pt_test for plotting later
    X_test, y_test, _, truth_pt_test, reco_pt_test, _, truth_mass_test, reco_mass_test = to_ML(data_test, class_labels, use_jets)
    save_test_data(out_dir, X_test, y_test, truth_pt_test, reco_pt_test, truth_mass_test, reco_mass_test, class_labels, use_jets)

    output_shape = y_train.shape[1:]
    if use_jets:
        X_train_constits, X_train_jets = X_train
        inputs =  {'constituent_inputs': X_train_constits, 'jet_inputs': X_train_jets}
        constituents_shape = X_train_constits.shape[1:]     #First dimension is batch size, input shape is NCONSTITUENTS x NFEATURES
        jets_shape = X_train_jets.shape[1:]    #First dimension is batch size, input shape is NJETS x NFEATURES
    else:
        inputs =  {'constituent_inputs': X_train}
        constituents_shape = X_train.shape[1:]    #First dimension is batch size, input shape is NCONSTITUENTS x NFEATURES
        jets_shape = None    #First dimension is batch size, input shape is NJETS x NFEATURES

    #Dynamically get the model
    try:
        model_func = getattr(models, model_name)
        model = model_func(constituents_shape, jets_shape, output_shape)  # Assuming the model function doesn't require additional arguments
        plot_model(model, to_file=f"{out_dir}/model.png", show_shapes=True, show_layer_names=True, show_layer_activations=True)
    except:
        raise ValueError(f"Model '{model_name}' is not defined in the 'models' module.")

    #Train it with a pruned model
    num_samples = inputs["constituent_inputs"].shape[0] * (1 - VALIDATION_SPLIT)
    pruned_model = prune_model(model, num_samples)

    #Now fit to the data
    callbacks = [tfmot.sparsity.keras.UpdatePruningStep(),
                 EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, start_from_epoch=1),
                 ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=3.125e-5)]

    from tagger.train.weights import flatten_weights, flatten_class_weights
    mlflow.autolog(disable=True)

    history = pruned_model.fit(
        inputs,
        {'jet_id_output': y_train}, #, 'pT_output': truth_pt_train, 'mass_output': truth_mass_train},
        sample_weight = {"jet_id_output": flatten_class_weights(y_train)},#, "pT_output": flatten_weights(reco_pt_train, 0, 2000, 61), "mass_output": flatten_weights(reco_mass_train, 0, 180, 61)},
        epochs = EPOCHS,
        batch_size = BATCH_SIZE,
        verbose = 2,
        validation_split = VALIDATION_SPLIT,
        callbacks = callbacks,
        shuffle = True
        )
    
    #Export the model
    model_export = pruned_model #tfmot.sparsity.keras.strip_pruning(pruned_model)

    export_path = os.path.join(out_dir, "model/saved_model.h5")
    model_export.save(export_path)
    print(f"Model saved to {export_path}")

    #Produce some basic plots with the training for diagnostics
    plot_path = os.path.join(out_dir, "plots/training")
    os.makedirs(plot_path, exist_ok=True)
    
    #Plot history
    loss_history(plot_path, history)

    return

if __name__ == "__main__":

    parser = ArgumentParser()

    #Making input arguments
    parser.add_argument('--make-data', action='store_true', help='Prepare the data if set.')
    parser.add_argument('-i','--input', default='./sc8_signal.root' , help = 'Path to input training data')
    parser.add_argument('-r','--ratio', default=1, type=float, help = 'Ratio (0-1) of the input data root file to process')
    parser.add_argument('-s','--step', default='10MB' , help = 'The maximum memory size to process input root file')
    parser.add_argument('-e','--extras', default='extra_fields', help= 'Which extra fields to add to output tuples, in pfcand_fields.yml')
    
    #Training argument
    parser.add_argument('-o','--output', default='output/baseline', help = 'Output model directory path, also save evaluation plots')
    parser.add_argument('-p','--percent', default=100, type=int, help = 'Percentage of how much processed data to train on')
    parser.add_argument('-m','--model', default='baseline', help = 'Model object name to train on')
    parser.add_argument('-n','--name', default='baseline', help = 'Model experiment name')
    parser.add_argument('-t','--tree', default='outnano/Jets', help = 'Tree within the ntuple containing the jets')
    parser.add_argument('--use-jets', action='store_true', help='Tell model if to expect jet-level features')

    #Basic ploting
    parser.add_argument('--plot-basic', action='store_true', help='Plot all the basic performance if set')

    args = parser.parse_args()

    mlflow.set_experiment(os.getenv('CI_COMMIT_REF_NAME'))

    #Either make data or start the training
    if args.make_data:
        make_data(infile=args.input, step_size=args.step, extras=args.extras, ratio=args.ratio, tree=args.tree) #Write to training_data/, can be specified using outdir, but keeping it simple here for now
    elif args.plot_basic:
        model_dir = args.output
        f = open("mlflow_run_id.txt", "r")
        run_id = (f.read())
        mlflow.get_experiment_by_name(os.getenv('CI_COMMIT_REF_NAME'))
        with mlflow.start_run(experiment_id=1,
                            run_name=args.name,
                            run_id=run_id # pass None to start a new run
                            ):

            #All the basic plots!
            results = basic(model_dir)
            for class_label in results.keys():
                mlflow.log_metric(class_label + ' ROC AUC',results[class_label])
            
    else:
        with mlflow.start_run(run_name=args.name) as run:
            mlflow.set_tag('gitlab.CI_JOB_ID', os.getenv('CI_JOB_ID'))
            mlflow.keras.autolog()
            train(args.output, args.percent, model_name=args.model, use_jets=args.use_jets)
            run_id = run.info.run_id
        sourceFile = open('mlflow_run_id.txt', 'w')
        print(run_id, end="", file = sourceFile)
        sourceFile.close()