"""
Here all the models are defined to be called in train.py
"""
import tensorflow as tf

# Qkeras
from qkeras.quantizers import quantized_bits, quantized_relu
from qkeras.qlayers import QDense, QActivation
from qkeras import QConv1D

# Keras
from tensorflow.keras.layers import BatchNormalization, Input, Dense, Activation, GlobalAveragePooling1D, GlobalMaxPooling1D, Flatten, Concatenate, Conv1D
from keras import activations

def baseline(constituents_shape, jets_shape, output_shape):
    print("Using baseline model!")
    #Initialize inputs
    constituent_input = Input(shape=constituents_shape, name='constituent_inputs')

    # #Main branch
    constituent_input_norm = BatchNormalization(name='norm_input')(constituent_input)
    
    #First Conv1D
    main = Conv1D(filters=32, kernel_size=1, name='Conv1D_1')(constituent_input_norm)
    main = Activation(activation=activations.relu, name='relu_1')(main)

    #Second Conv1D
    main = Conv1D(filters=16, kernel_size=1, name='Conv1D_2')(main)
    main = Activation(activation=activations.relu, name='relu_2')(main)

    # Linear activation to change HLS bitwidth to fix overflow in AveragePooling
    # main = Activation(activation='quantized_bits(18,8)', name = 'act_pool')(main)
    main = GlobalAveragePooling1D(name='avgpool')(main)

    inputs = {"constituent_inputs": constituent_input}
    if jets_shape is not None:
        print("Using jet features in the model!")
        # If jet features are provided, concatenate them
        jet_input = Input(shape=jets_shape, name='jet_inputs') # Shape is (n_jet_features,)
        norm_jet_input = BatchNormalization(name='norm_jet_input')(jet_input)
        main = Concatenate(name='combine_features')([main, norm_jet_input]) # Shape: (batch_size, 10 + n_jet_features)
        inputs["jet_inputs"] = jet_input

    #Now split into jet ID, pt regression and mass regression

    #jetID branch, 3 layer MLP
    jet_id = Dense(32, name='Dense_1_jetID')(main)
    jet_id = Activation(activation=activations.relu, name='relu_1_jetID')(jet_id)

    jet_id = Dense(16, name='Dense_2_jetID')(jet_id)
    jet_id = Activation(activation=activations.relu, name='relu_2_jetID')(jet_id)

    jet_id = Dense(output_shape[0], name='Dense_3_jetID')(jet_id)
    jet_id = Activation('softmax', name='jet_id_output')(jet_id)

    #pT regression branch
    pt_regress = Dense(16, name='Dense_1_pT')(main)
    pt_regress = Activation(activation=activations.relu, name='relu_1_pt')(pt_regress)
    pt_regress = Dense(8, name='Dense_2_pT')(pt_regress)
    pt_regress = Activation(activation=activations.relu, name='relu_2_pt')(pt_regress)
    pt_regress = Dense(1, name='pT_output')(pt_regress)
    
    # mass regression branch
    mass_regress = Dense(16, name='Dense_1_mass')(main)
    mass_regress = Activation(activation=activations.relu, name='relu_1_mass')(mass_regress)
    mass_regress = Dense(8, name='Dense_2_mass')(mass_regress)
    mass_regress = Activation(activation=activations.relu, name='relu_2_mass')(mass_regress)
    mass_regress = Dense(1, name='mass_output')(mass_regress)

    #Define the model using both branches
    model = tf.keras.Model(inputs=inputs, outputs = [jet_id, pt_regress, mass_regress])

    print(model.summary())

    return model


def qbaseline(constituents_shape, jets_shape, output_shape, bits=9, bits_int=2, alpha_val=1):

    # Define a dictionary for common arguments
    common_args = {
        'kernel_quantizer': quantized_bits(bits, bits_int, alpha=alpha_val),
        'bias_quantizer': quantized_bits(bits, bits_int, alpha=alpha_val),
        'kernel_initializer': 'lecun_uniform'
    }

    #Initialize inputs
    constituent_input = Input(shape=constituents_shape, name='constituent_inputs')

    # #Main branch
    constituent_input_norm = BatchNormalization(name='norm_input')(constituent_input)
    
    #First Conv1D
    main = QConv1D(filters=32, kernel_size=1, name='Conv1D_1', **common_args)(constituent_input_norm)
    main = QActivation(activation=quantized_relu(bits), name='relu_1')(main)

    #Second Conv1D
    main = QConv1D(filters=16, kernel_size=1, name='Conv1D_2', **common_args)(main)
    main = QActivation(activation=quantized_relu(bits), name='relu_2')(main)

    # Linear activation to change HLS bitwidth to fix overflow in AveragePooling
    main = QActivation(activation='quantized_bits(18,8)', name = 'act_pool')(main)
    main = GlobalAveragePooling1D(name='avgpool')(main)

    inputs = {"constituent_inputs": constituent_input}
    if jets_shape is not None:
        print("Using jet features in the model!")
        # If jet features are provided, concatenate them
        jet_input = Input(shape=jets_shape, name='jet_inputs') # Shape is (n_jet_features,)
        norm_jet_input = BatchNormalization(name='norm_jet_input')(jet_input)
        main = Concatenate(name='combine_features')([main, norm_jet_input]) # Shape: (batch_size, 10 + n_jet_features)
        inputs["jet_inputs"] = jet_input

    #Now split into jet ID, pt regression and mass regression

    #jetID branch, 3 layer MLP
    jet_id = QDense(32, name='Dense_1_jetID', **common_args)(main)
    jet_id = QActivation(activation=quantized_relu(bits), name='relu_1_jetID')(jet_id)

    jet_id = QDense(16, name='Dense_2_jetID', **common_args)(jet_id)
    jet_id = QActivation(activation=quantized_relu(bits), name='relu_2_jetID')(jet_id)

    jet_id = QDense(output_shape[0], name='Dense_3_jetID', **common_args)(jet_id)
    jet_id = Activation('softmax', name='jet_id_output')(jet_id)

    #pT regression branch
    pt_regress = QDense(16, name='Dense_1_pT', **common_args)(main)
    pt_regress = QActivation(activation=quantized_relu(bits), name='relu_1_pt')(pt_regress)
    pt_regress = QDense(8, name='Dense_2_pT', **common_args)(pt_regress)
    pt_regress = QActivation(activation=quantized_relu(bits), name='relu_2_pt')(pt_regress)
    pt_regress = QDense(1, name='pT_output',
                        kernel_quantizer=quantized_bits(16, 6, alpha=alpha_val),
                        bias_quantizer=quantized_bits(16, 6, alpha=alpha_val),
                        kernel_initializer='lecun_uniform')(pt_regress)
    
    # mass regression branch
    mass_regress = QDense(16, name='Dense_1_mass', **common_args)(main)
    mass_regress = QActivation(activation=quantized_relu(bits), name='relu_1_mass')(mass_regress)
    mass_regress = QDense(8, name='Dense_2_mass', **common_args)(mass_regress)
    mass_regress = QActivation(activation=quantized_relu(bits), name='relu_2_mass')(mass_regress)
    mass_regress = QDense(1, name='mass_output',
                        kernel_quantizer=quantized_bits(16, 6, alpha=alpha_val),
                        bias_quantizer=quantized_bits(16, 6, alpha=alpha_val),
                        kernel_initializer='lecun_uniform')(mass_regress)

    #Define the model using both branches
    model = tf.keras.Model(inputs=inputs, outputs = [jet_id, pt_regress, mass_regress])

    print(model.summary())

    return model