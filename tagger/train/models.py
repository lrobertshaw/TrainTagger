"""
Here all the models are defined to be called in train.py
"""
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Input, Activation, GlobalAveragePooling1D, Flatten

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
    inputs = tf.keras.layers.Input(shape=inputs_shape, name='model_input')

    #Main branch
    main = BatchNormalization(name='norm_input')(inputs)
    
    #First Conv1D
    main = QConv1D(filters=10, kernel_size=1, name='Conv1D_1', **common_args)(main)
    main = QActivation(activation=quantized_relu(bits), name='relu_1')(main)

    #Second Conv1D
    main = QConv1D(filters=10, kernel_size=1, name='Conv1D_2', **common_args)(main)
    main = QActivation(activation=quantized_relu(bits), name='relu_2')(main)

    # main = QActivation(activation='quantized_bits(18,8)', name = 'act_pool')(main)
    # main = GlobalAveragePooling1D(name='avgpool')(main)

    # --- Replacement for GlobalAveragePooling1D ---
    # Optional: Apply the linear activation if still needed for bitwidth before dense
    # You might need to adjust the bits here depending on the Flatten output size
    # main = QActivation(activation='quantized_bits(18,8)', name='act_pre_flatten')(main)

    # Flatten the output of the conv layers
    # Output shape will be (batch_size, num_constituents * features_after_conv)
    # mass_regress = Flatten(name='flatten_constituents')(main)

    # Add one or more shared Dense layers to process the flattened features
    mass_regress = QDense(64, name='Dense_shared_1', **common_args)(main) # Example size, tune this
    mass_regress = QActivation(activation=quantized_relu(bits), name='relu_shared_1')(mass_regress)
    # You could add another shared Dense layer here if needed
    mass_regress = QDense(32, name='Dense_shared_2', **common_args)(mass_regress)
    mass_regress = QActivation(activation=quantized_relu(bits), name='relu_shared_2')(mass_regress)
    # --- End Replacement ---

    # pT regression head
    # pt_regress = QActivation(activation='quantized_bits(18,8)', name = 'act_pool')(main)
    # pt_regress = GlobalAveragePooling1D(name='avgpool')(pt_regress)
    # pt_regress = QDense(10, name='Dense_1_pT', **common_args)(pt_regress)
    # pt_regress = QActivation(activation=quantized_relu(bits), name='relu_1_pt')(pt_regress)
    # pt_regress = QDense(1, name='pT_output',
    #                     kernel_quantizer=quantized_bits(16, 6, alpha=alpha_val),
    #                     bias_quantizer=quantized_bits(16, 6, alpha=alpha_val),
    #                     kernel_initializer='lecun_uniform')(pt_regress)
    
    # mass regression head
    mass_regress = QDense(16, name='Dense_1_mass', **common_args)(mass_regress)
    mass_regress = QActivation(activation=quantized_relu(bits), name='relu_1_mass')(mass_regress)
    mass_regress = QDense(1, name='mass_output',
                        kernel_quantizer=quantized_bits(16, 6, alpha=alpha_val),
                        bias_quantizer=quantized_bits(16, 6, alpha=alpha_val),
                        kernel_initializer='lecun_uniform')(mass_regress)

    #Define the model using both branches
    model = tf.keras.Model(inputs = inputs, outputs = [mass_regress])   # pt_regress

    print(model.summary())

    return model
