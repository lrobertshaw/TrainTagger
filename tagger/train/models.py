"""
Here all the models are defined to be called in train.py
"""
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Input, Activation, GlobalAveragePooling1D, Flatten, Concatenate

# Qkeras
from qkeras.quantizers import quantized_bits, quantized_relu
from qkeras.qlayers import QDense, QActivation
from qkeras import QConv1D


def baseline(inputs_shape, output_shape, bits=9, bits_int=2, alpha_val=1):

    # Define a dictionary for common arguments
    common_args = {
        'kernel_quantizer': quantized_bits(bits, bits_int, alpha=alpha_val),
        'bias_quantizer': quantized_bits(bits, bits_int, alpha=alpha_val),
        'kernel_initializer': 'lecun_uniform'
    }

    #Initialize inputs
    inputs = tf.keras.layers.Input(shape=inputs_shape, name='constituent_level_inputs')

    #Main branch
    main = BatchNormalization(name='norm_input')(inputs)
    
    #First Conv1D
    main = QConv1D(filters=10, kernel_size=1, name='Conv1D_1', **common_args)(main)
    main = QActivation(activation=quantized_relu(bits), name='relu_1')(main)

    #Second Conv1D
    main = QConv1D(filters=10, kernel_size=1, name='Conv1D_2', **common_args)(main)
    main = QActivation(activation=quantized_relu(bits), name='relu_2')(main)

    # Linear activation to change HLS bitwidth to fix overflow in AveragePooling
    main = QActivation(activation='quantized_bits(18,8)', name = 'act_pool')(main)
    main = GlobalAveragePooling1D(name='avgpool')(main)


    #Now split into jet pt and mass regression

    #pT regression branch
    pt_regress = QDense(10, name='Dense_1_pT', **common_args)(main)
    pt_regress = QActivation(activation=quantized_relu(bits), name='relu_1_pt')(pt_regress)
    pt_regress = QDense(1, name='pT_output',
                        kernel_quantizer=quantized_bits(16, 6, alpha=alpha_val),
                        bias_quantizer=quantized_bits(16, 6, alpha=alpha_val),
                        kernel_initializer='lecun_uniform')(pt_regress)
    
    # mass regression branch
    mass_regress = QDense(10, name='Dense_1_mass', **common_args)(main)
    mass_regress = QActivation(activation=quantized_relu(bits), name='relu_1_mass')(mass_regress)
    mass_regress = QDense(1, name='mass_output',
                        kernel_quantizer=quantized_bits(16, 6, alpha=alpha_val),
                        bias_quantizer=quantized_bits(16, 6, alpha=alpha_val),
                        kernel_initializer='lecun_uniform')(mass_regress)

    #Define the model using both branches
    model = tf.keras.Model(inputs = inputs, outputs = [pt_regress, mass_regress])

    print(model.summary())

    return model



def baseline_with_jet_features(constituent_shape, n_jet_features, output_shape, bits=9, bits_int=2, alpha_val=1):
    """
    Builds a Keras model with constituent and jet-level inputs.

    Args:
        constituent_shape: Tuple, shape of the constituent input (n_constituents, n_features_per_constituent).
        n_jet_features: Integer, number of jet-level features (e.g., 3 for pt, eta, phi).
        output_shape: Tuple, shape of the output (not directly used for defining outputs here, but kept for consistency).
        bits: Integer, quantization bits.
        bits_int: Integer, integer part bits for quantization.
        alpha_val: Float, alpha value for quantization.

    Returns:
        tf.keras.Model: The compiled Keras model.
    """

    common_args = {
        'kernel_quantizer': quantized_bits(bits, bits_int, alpha=alpha_val),
        'bias_quantizer': quantized_bits(bits, bits_int, alpha=alpha_val),
        'kernel_initializer': 'lecun_uniform'
    }

    # --- Define Two Inputs ---
    # Input for constituent data (sequence)
    constituent_input = Input(shape=constituent_shape, name='constituents_input')
    # Input for jet-level data (vector)
    jet_input = Input(shape=(n_jet_features,), name='jet_input') # Shape is (n_jet_features,)

    # --- Process Constituent Input (Original Main Branch) ---
    # Normalization applied only to constituents
    main = BatchNormalization(name='norm_constituent_input')(constituent_input)

    # First Conv1D on constituents
    main = QConv1D(filters=10, kernel_size=1, name='Conv1D_1', **common_args)(main)
    main = QActivation(activation=quantized_relu(bits), name='relu_1')(main)

    # Second Conv1D on constituents
    main = QConv1D(filters=10, kernel_size=1, name='Conv1D_2', **common_args)(main)
    main = QActivation(activation=quantized_relu(bits), name='relu_2')(main)

    # Linear activation before pooling
    main = QActivation(activation='quantized_bits(18,8)', name='act_pool')(main)
    # Pool constituent features
    constituent_features = GlobalAveragePooling1D(name='avgpool')(main) # Output shape: (batch_size, 10)

    # --- Optional: Normalize Jet Input ---
    # You might want to normalize the jet features as well, depending on their scale
    # normalized_jet_input = BatchNormalization(name='norm_jet_input')(jet_input)
    # If normalized, use normalized_jet_input in Concatenate below

    # --- Combine Features ---
    # Concatenate the pooled constituent features and the jet-level features
    # Ensure the jet_input is used here (or normalized_jet_input if you add normalization)
    combined_features = Concatenate(name='combine_features')([constituent_features, jet_input]) # Shape: (batch_size, 10 + n_jet_features)

    # --- Regression Heads (Applied to Combined Features) ---

    # pT regression branch
    pt_branch = QDense(10, name='Dense_1_pT', **common_args)(combined_features) # Takes combined features
    pt_branch = QActivation(activation=quantized_relu(bits), name='relu_1_pt')(pt_branch)
    pt_regress = QDense(1, name='pT_output',
                        kernel_quantizer=quantized_bits(16, 6, alpha=alpha_val),
                        bias_quantizer=quantized_bits(16, 6, alpha=alpha_val),
                        kernel_initializer='lecun_uniform')(pt_branch)

    # mass regression branch
    mass_branch = QDense(10, name='Dense_1_mass', **common_args)(combined_features) # Takes combined features
    mass_branch = QActivation(activation=quantized_relu(bits), name='relu_1_mass')(mass_branch)
    mass_regress = QDense(1, name='mass_output',
                          kernel_quantizer=quantized_bits(16, 6, alpha=alpha_val),
                          bias_quantizer=quantized_bits(16, 6, alpha=alpha_val),
                          kernel_initializer='lecun_uniform')(mass_branch)

    # --- Define the Model with Multiple Inputs ---
    # The 'inputs' argument is now a list containing both input layers
    model = tf.keras.Model(inputs=[constituent_input, jet_input],
                  outputs=[pt_regress, mass_regress],
                  name="jet_regression_with_globals")

    return model