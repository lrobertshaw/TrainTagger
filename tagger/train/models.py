"""
Here all the models are defined to be called in train.py
"""
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Input, Activation, GlobalAveragePooling1D, Flatten, Concatenate

# Qkeras
from qkeras.quantizers import quantized_bits, quantized_relu
from qkeras.qlayers import QDense, QActivation
from qkeras import QConv1D


def baseline(inputs_shape, bits=9, bits_int=2, alpha_val=1):
    print('Using model: "baseline"')

    # Define a dictionary for common arguments
    common_args = {
        'kernel_quantizer': quantized_bits(bits, bits_int, alpha=alpha_val),
        'bias_quantizer': quantized_bits(bits, bits_int, alpha=alpha_val),
        'kernel_initializer': 'lecun_uniform'
    }

    #Initialize inputs
    inputs = tf.keras.layers.Input(shape=inputs_shape, name='constituent_inputs')

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


def baseline_with_jet_features(input_shape, bits=9, bits_int=2, alpha_val=1):
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
    print('Using model: "baseline_with_jet_features"')

    try:
        constituent_shape, n_jet_features = input_shape
    except ValueError:
        raise ValueError("This model uses constituent AND jet level features, but only one input shape was provided. "
                         "Please provide a tuple of (constituent_shape, n_jet_features) or use a different model.")

    common_args = {
        'kernel_quantizer': quantized_bits(bits, bits_int, alpha=alpha_val),
        'bias_quantizer': quantized_bits(bits, bits_int, alpha=alpha_val),
        'kernel_initializer': 'lecun_uniform'
    }

    # --- Define Two Inputs ---
    # Input for constituent data (sequence)
    constituent_input = Input(shape=constituent_shape, name='constituent_inputs')

    # --- Process Constituent Input (Original Main Branch) ---
    # Normalization applied only to constituents
    main = BatchNormalization(name='norm_constituent_input')(constituent_input)

    # First Conv1D on constituents
    main = QConv1D(filters=16, kernel_size=1, name='Conv1D_1', **common_args)(main)
    main = QActivation(activation=quantized_relu(bits), name='relu_1')(main)

    # Second Conv1D on constituents
    main = QConv1D(filters=32, kernel_size=1, name='Conv1D_2', **common_args)(main)
    main = QActivation(activation=quantized_relu(bits), name='relu_2')(main)

    main = QConv1D(filters=64, kernel_size=1, name='Conv1D_3', **common_args)(main)
    main = QActivation(activation=quantized_relu(bits), name='relu_3')(main)

    # Linear activation before pooling
    main = QActivation(activation='quantized_bits(18,8)', name='act_pool')(main)
    # Pool constituent features
    constituent_features = GlobalAveragePooling1D(name='avgpool')(main) # Output shape: (batch_size, 10)

    jet_input = Input(shape=n_jet_features, name='jet_inputs') # Shape is (n_jet_features,)
    norm_jet_input = BatchNormalization(name='norm_jet_input')(jet_input)
    combined_features = Concatenate(name='combine_features')([constituent_features, norm_jet_input]) # Shape: (batch_size, 10 + n_jet_features)

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
    # model = tf.keras.Model(inputs=[constituent_input, jet_input], outputs=[mass_regress], name="jet_regression_with_globals")
    model = tf.keras.Model(inputs=[constituent_input, jet_input], outputs=[pt_regress, mass_regress], name="jet_regression_with_globals")

    return model


def complex_with_jet_features(input_shape, bits=9, bits_int=2, alpha_val=1):
    """
    Builds a more complex Keras model for jet pT and mass regression
    with attention, residual connections, and earlier interaction.

    Args:
        constituent_shape: Tuple (n_constituents, n_features_per_constituent).
        n_jet_features: Integer, number of jet-level features.
        bits: Integer, quantization bits.
        bits_int: Integer, integer part bits for quantization.
        alpha_val: Float, alpha value for quantization.

    Returns:
        tf.keras.Model: The compiled Keras model.
    """

    print('Using model: "complex_with_jet_features"')

    from tensorflow.keras.layers import (
        Input, BatchNormalization, GlobalAveragePooling1D, Concatenate, Dense,
        Conv1D, Activation, Add, Multiply, Lambda, RepeatVector, TimeDistributed
    )
    import tensorflow.keras.backend as K

    constituent_shape, n_jet_features = input_shape
    common_args_q = {
        'kernel_quantizer': quantized_bits(bits, bits_int, alpha=alpha_val),
        'bias_quantizer': quantized_bits(bits, bits_int, alpha=alpha_val),
        'kernel_initializer': 'lecun_uniform'
    }
    dense_output_args_q = {
        'kernel_quantizer': quantized_bits(16, 6, alpha=alpha_val),
        'bias_quantizer': quantized_bits(16, 6, alpha=alpha_val),
        'kernel_initializer': 'lecun_uniform'
    }
    n_constituents = constituent_shape[0]
    n_constituent_features = constituent_shape[1]

    # === Inputs ===
    constituent_input = Input(shape=constituent_shape, name='constituent_input')
    jet_input = Input(shape=(n_jet_features,), name='jet_input')

    # === Early Interaction ===
    # Repeat jet features for each constituent and concatenate
    # Shape jet_input: (batch, n_jet_features)
    # Shape repeated_jet_input: (batch, n_constituents, n_jet_features)
    repeated_jet_input = RepeatVector(n_constituents, name='repeat_jet_feats')(jet_input)
    # Shape constituent_input: (batch, n_constituents, n_constituent_features)
    # Shape combined_constituent_input: (batch, n_constituents, n_constituent_features + n_jet_features)
    combined_constituent_input = Concatenate(axis=-1, name='concat_const_jet_early')([constituent_input, repeated_jet_input])

    # === Initial Processing ===
    # Normalize the combined constituent features
    x = BatchNormalization(name='norm_combined_input')(combined_constituent_input)
    # Initial Conv to adjust dimensionality (e.g., to 32 filters)
    initial_filters = 32
    x = QConv1D(filters=initial_filters, kernel_size=1, name='Conv1D_Initial', **common_args_q)(x)
    x = QActivation(activation=quantized_relu(bits), name='relu_initial')(x)

    # === Deeper Convolutional Blocks with Residual Connections ===
    def res_block(input_tensor, filters, block_num):
        # Main path
        res_x = QConv1D(filters=filters, kernel_size=1, padding='same', name=f'ResConv1_{block_num}a', **common_args_q)(input_tensor)
        res_x = QActivation(activation=quantized_relu(bits), name=f'ResRelu_{block_num}a')(res_x)
        res_x = QConv1D(filters=filters, kernel_size=1, padding='same', name=f'ResConv1_{block_num}b', **common_args_q)(res_x)
        # Shortcut path
        # If input filters match output filters, use identity, else use a 1x1 Conv to match
        if K.int_shape(input_tensor)[-1] == filters:
            shortcut = input_tensor
        else:
            shortcut = QConv1D(filters=filters, kernel_size=1, name=f'ResShortcutConv_{block_num}', **common_args_q)(input_tensor)
            shortcut = BatchNormalization(name=f'ResShortcutBN_{block_num}')(shortcut) # Optional BN on shortcut

        # Add shortcut to main path
        res_x = Add(name=f'ResAdd_{block_num}')([shortcut, res_x])
        res_x = QActivation(activation=quantized_relu(bits), name=f'ResRelu_{block_num}b')(res_x)
        return res_x

    # Add a few residual blocks
    filters_block1 = 32
    filters_block2 = 64
    x = res_block(x, filters_block1, 1)
    x = res_block(x, filters_block1, 2)
    x = res_block(x, filters_block2, 3) # Increase filters

    # === Attention Mechanism ===
    # Calculate attention weights for each constituent
    # Use a Conv1D with 1 filter and softmax activation over the constituent dimension
    attention_filters = 1 # Should be 1 to get a single weight per constituent
    # Shape x: (batch, n_constituents, filters_block2)
    # Shape attention_weights: (batch, n_constituents, 1)
    attention_weights = QConv1D(filters=attention_filters, kernel_size=1, activation='softmax', name='Attention_SoftmaxConv', **common_args_q)(x) # Using standard softmax here

    # Apply attention weights to the features (element-wise multiplication)
    # Shape attended_features: (batch, n_constituents, filters_block2)
    attended_features = Multiply(name='Attention_Multiply')([x, attention_weights])

    # === Pooling ===
    # Pool the *attended* features
    # Shape pooled_constituent_features: (batch, filters_block2)
    pooled_constituent_features = GlobalAveragePooling1D(name='Attention_AvgPool')(attended_features)

    # === Combine Pooled Features with Global Jet Features ===
    # Concatenate the attention-pooled constituent features and the original jet-level features
    # Shape combined_late_features: (batch, filters_block2 + n_jet_features)
    combined_late_features = Concatenate(name='combine_pooled_jet_late')([pooled_constituent_features, jet_input])

    # === Final Dense Heads (Deeper/Wider) ===
    dense_units_1 = 64 # Increased units
    dense_units_2 = 32

    # --- pT regression branch ---
    pt_branch = QDense(dense_units_1, name='Dense_1_pT', **common_args_q)(combined_late_features)
    pt_branch = QActivation(activation=quantized_relu(bits), name='relu_1_pt')(pt_branch)
    pt_branch = QDense(dense_units_2, name='Dense_2_pT', **common_args_q)(pt_branch)
    pt_branch = QActivation(activation=quantized_relu(bits), name='relu_2_pt')(pt_branch)
    pt_regress = QDense(1, name='pT_output', **dense_output_args_q)(pt_branch)

    # --- mass regression branch ---
    mass_branch = QDense(dense_units_1, name='Dense_1_mass', **common_args_q)(combined_late_features)
    mass_branch = QActivation(activation=quantized_relu(bits), name='relu_1_mass')(mass_branch)
    mass_branch = QDense(dense_units_2, name='Dense_2_mass', **common_args_q)(mass_branch)
    mass_branch = QActivation(activation=quantized_relu(bits), name='relu_2_mass')(mass_branch)
    mass_regress = QDense(1, name='mass_output', **dense_output_args_q)(mass_branch)

    # === Define Model ===
    model = tf.keras.Model(inputs=[constituent_input, jet_input],
                  outputs=[pt_regress, mass_regress],
                  name="complex_with_jet_features")

    return model