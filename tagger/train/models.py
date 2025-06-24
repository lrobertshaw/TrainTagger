"""
Here all the models are defined to be called in train.py
"""
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Input, Activation, GlobalAveragePooling1D, GlobalMaxPooling1D, Flatten, Concatenate

import numpy as np
import itertools
from tensorflow import keras
from tensorflow.keras import layers as KL

# Qkeras
from qkeras.quantizers import quantized_bits, quantized_relu
from qkeras.qlayers import QDense, QActivation#, QBatchNormalization
from qkeras import QConv1D, QBatchNormalization

from tensorflow.keras.layers import BatchNormalization, Input, Dense, Activation, GlobalAveragePooling1D, GlobalMaxPooling1D, Flatten, Concatenate, Conv1D
from keras import activations

def baseline(constituents_shape, jets_shape=None):
    print('Using model: "baseline"')
    if jets_shape is not None:
        print("Using jet features in the model!")
    else: 
        print("No jet features used in the model!")


    # #Initialize inputs
    constituent_input = Input(shape=constituents_shape, name='constituent_inputs')

    # #Main branch
    constituent_input_norm = BatchNormalization(name='norm_input')(constituent_input)
    
    # #First Conv1D
    main = Conv1D(filters=10, kernel_size=1, name='Conv1D_1')(constituent_input_norm)
    main = Activation(activation=activations.relu, name='relu_1')(main)

    # #Second Conv1D
    main = Conv1D(filters=10, kernel_size=1, name='Conv1D_2')(main)
    main = Activation(activation=activations.relu, name='relu_2')(main)

    # # Linear activation to change HLS bitwidth to fix overflow in AveragePooling
    # main = Activation(activation=activations.relu, name = 'act_pool')(main)
    main = GlobalAveragePooling1D(name='avgpool')(main)
    
    inputs = {"constituent_inputs": constituent_input}
    if jets_shape is not None:
        print("Using jet features in the model!")
        # If jet features are provided, concatenate them
        jet_input = Input(shape=jets_shape, name='jet_inputs') # Shape is (n_jet_features,)
        jet_input_norm = BatchNormalization(name='norm_jet_input')(jet_input)
        main = Concatenate(name='combine_features')([main, jet_input_norm]) # Shape: (batch_size, 10 + n_jet_features)
        inputs["jet_inputs"] = jet_input

    # pt regression branch
    pt_regress = Dense(10, name='Dense_1_pt')(main)
    pt_regress = Activation(activation=activations.relu, name='relu_1_pt')(pt_regress)
    # pt_regress = Dense(8, name='Dense_2_pt')(pt_regress)
    # pt_regress = Activation(activation=activations.relu, name='relu_2_pt')(pt_regress)
    pt_regress = Dense(1, name='pT_output')(pt_regress)
    
    # mass regression branch
    mass_regress = Dense(10, name='Dense_1_mass')(main)
    mass_regress = Activation(activation=activations.relu, name='relu_1_mass')(mass_regress)
    # mass_regress = Dense(8, name='Dense_2_mass')(mass_regress)
    # mass_regress = Activation(activation=activations.relu, name='relu_2_mass')(mass_regress)
    mass_regress = Dense(1, name='mass_output')(mass_regress)

    model = tf.keras.Model(inputs=inputs, outputs=[pt_regress, mass_regress], name="baseline")
    print(model.summary())

    return model


def qbaseline(constituents_shape, jets_shape=None, bits=9, bits_int=2, alpha_val="auto"):
    print('Using model: "quantised baseline"')
    common_args = {
        'kernel_quantizer': quantized_bits(bits, bits_int, alpha=alpha_val),
        'bias_quantizer': quantized_bits(bits, bits_int, alpha=alpha_val),
        'kernel_initializer': 'lecun_uniform'
    }

    #Initialize inputs
    constituent_input = tf.keras.layers.Input(shape=constituents_shape, name='constituent_inputs')

    #Main branch
    main = QBatchNormalization(name='norm_input')(constituent_input)
    
    #First Conv1D
    main = QConv1D(filters=10, kernel_size=1, name='Conv1D_1', **common_args)(main)
    main = QActivation(activation=quantized_relu(bits), name='relu_1')(main)

    #Second Conv1D
    main = QConv1D(filters=10, kernel_size=1, name='Conv1D_2', **common_args)(main)
    main = QActivation(activation=quantized_relu(bits), name='relu_2')(main)

    # Linear activation to change HLS bitwidth to fix overflow in AveragePooling
    main = QActivation(activation='quantized_bits(18, 8)', name = 'act_pool')(main)
    main = GlobalAveragePooling1D(name='avgpool')(main)

    # pt regression branch
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
    model = tf.keras.Model(inputs=constituent_input, outputs=[pt_regress, mass_regress], name="baseline")
    print(model.summary())

    return model


class NodeEdgeProjection(KL.Layer):
    """Layer that build the adjacency matrix for the interaction network graph.

    Attributes:
        receiving: Whether we are building the receiver (True) or sender (False)
            adjency matrix.
        node_to_edge: Whether the projection happens from nodes to edges (True) or
            the edge matrix gets projected into the nodes (False).
    """

    def __init__(self, receiving: bool = True, node_to_edge: bool = True, **kwargs):
        super().__init__(**kwargs)
        self._receiving = receiving
        self._node_to_edge = node_to_edge

    def build(self, input_shape: tuple):
        if self._node_to_edge:
            self._n_nodes = input_shape[-2]
            self._n_edges = self._n_nodes * (self._n_nodes - 1)
        else:
            self._n_edges = input_shape[-2]
            self._n_nodes = int((np.sqrt(4 * self._n_edges + 1) + 1) / 2)

        self._adjacency_matrix = self._assign_adjacency_matrix()

    def _assign_adjacency_matrix(self):
        receiver_sender_list = itertools.permutations(range(self._n_nodes), r=2)
        if self._node_to_edge:
            shape, adjacency_matrix = self._assign_node_to_edge(receiver_sender_list)
        else:
            shape, adjacency_matrix = self._assign_edge_to_node(receiver_sender_list)

        return tf.Variable(
            initial_value=adjacency_matrix,
            name="adjacency_matrix",
            dtype="float32",
            shape=shape,
            trainable=False,
        )

    def _assign_node_to_edge(self, receiver_sender_list: list):
        shape = (1, self._n_edges, self._n_nodes)
        adjacency_matrix = np.zeros(shape, dtype=float)
        for i, (r, s) in enumerate(receiver_sender_list):
            if self._receiving:
                adjacency_matrix[0, i, r] = 1
            else:
                adjacency_matrix[0, i, s] = 1

        return shape, adjacency_matrix

    def _assign_edge_to_node(self, receiver_sender_list: list):
        shape = (1, self._n_nodes, self._n_edges)
        adjacency_matrix = np.zeros(shape, dtype=float)
        for i, (r, s) in enumerate(receiver_sender_list):
            if self._receiving:
                adjacency_matrix[0, r, i] = 1
            else:
                adjacency_matrix[0, s, i] = 1

        return shape, adjacency_matrix

    def call(self, inputs):
        return tf.matmul(self._adjacency_matrix, inputs)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "receiving": self._receiving,
                "node_to_edge": self._node_to_edge,
            }
        )
        return config
    def get_prunable_weights(self):
    # Return the adjacency matrix to be pruned
        return [self._adjacency_matrix]

def IntNet(constituents_shape, jets_shape, effects_layers=[16, 8], objects_layers=[32, 16, 8], 
           classifier_layers=[128], activ="relu", aggreg="mean", bits=9, bits_int=2, alpha_val=1):
    """
    Functional version of IntNetQuantised.

    Args:
        inputs_shape (tuple): Shape of the input data.
        output_dim (int): Output dimension (number of classes, default=5).
        effects_layers (list): Number of nodes for each layer in the effects MLP.
        objects_layers (list): Number of nodes for each layer in the objects MLP.
        classifier_layers (list): Number of nodes for each layer in the classifier MLP.
        activ (str): Activation function.
        aggreg (str): Aggregation type ("mean" or "max").
        nbits (int): Number of bits for quantization.

    Returns:
        keras.Model: Compiled Interaction Network Model.
    """
    common_args = {
        'kernel_quantizer': quantized_bits(bits, bits_int, alpha=alpha_val),
        'bias_quantizer': quantized_bits(bits, bits_int, alpha=alpha_val),
        'kernel_initializer': 'lecun_uniform'
    }
    # Define input layer
    constituent_inputs = tf.keras.layers.Input(shape=constituents_shape, name='constituent_inputs')

    constituent_inputs = BatchNormalization(name='norm_input')(constituent_inputs)

    # Define quantization settings
    #quant = format_quantiser(nbits)
    activation_fn = format_qactivation(activ, bits)

    # Node-to-Edge projections
    receiver_matrix_proj = NodeEdgeProjection(name="receiver_matrix", receiving=True, node_to_edge=True)(constituent_inputs)
    sender_matrix_proj = NodeEdgeProjection(name="sender_matrix", receiving=False, node_to_edge=True)(constituent_inputs)

    # Concatenate sender and receiver projections
    input_effects = KL.Concatenate(axis=-1, name="concat_eff")([receiver_matrix_proj, sender_matrix_proj])

    # Build Effects Network (MLP)
    x = input_effects
    for layer_size in effects_layers:
        x = QConv1D(layer_size, kernel_size=1, **common_args)(x)
        x = QActivation(activation_fn)(x)
    effects_output = NodeEdgeProjection(name="prj_effects", receiving=True, node_to_edge=False)(x)

    # Concatenate effects with original inputs
    input_objects = KL.Concatenate(axis=-1, name="concat_obj")([constituent_inputs, effects_output])

    # Build Objects Network (MLP)
    x = input_objects
    for layer_size in objects_layers:
        x = QConv1D(layer_size, kernel_size=1, **common_args)(x)
        x = QActivation(activation_fn)(x)

    # Apply aggregation (mean or max)
    agg=choose_aggregator(aggreg)
    x=agg(x)

    # --- Optional Jet-Level Input ---
    inputs = {"constituent_inputs": constituent_inputs}
    if jets_shape is not None:
        jet_inputs = tf.keras.layers.Input(shape=jets_shape, name='jet_inputs')
        norm_jet_input = BatchNormalization(name='norm_jet_input')(jet_inputs)
        x = tf.keras.layers.Concatenate(name="combine_constituents_and_jet")([x, norm_jet_input])
        inputs["jet_inputs"] = jet_inputs

    # Build Classifier Network (MLP)
    #x = aggreg_output
    for layer_size in classifier_layers:
        x = QDense(layer_size, **common_args)(x)
        x = QActivation(activation_fn)(x)

    # Final output layer

    # pt_regress = QDense(8, name="pt_dense_1",**common_args)(x)
    # pt_regress = QActivation(activation=quantized_relu(9), name='relu_1_pt')(pt_regress)
    # pt_regress = QDense(1, name="pT_output",kernel_quantizer=quantized_bits(16, 6, alpha=alpha_val),
    #                            bias_quantizer=quantized_bits(16, 6, alpha=alpha_val),
    #                            kernel_initializer='lecun_uniform'
    #                            )(pt_regress)
    
    mass_regress = QDense(16, name="mass_dense_1",**common_args)(x)
    mass_regress = QActivation(activation=quantized_relu(9), name='relu_1_mass')(mass_regress)
    mass_regress = QDense(8, name="mass_dense_2",**common_args)(mass_regress)
    mass_regress = QActivation(activation=quantized_relu(9), name='relu_2_mass')(mass_regress)
    mass_regress = QDense(1, name="mass_output", kernel_quantizer=quantized_bits(16, 6, alpha=alpha_val),
                                 bias_quantizer=quantized_bits(16, 6, alpha=alpha_val),
                                 kernel_initializer='lecun_uniform'
                                 )(mass_regress)
    # Create model
    model = keras.Model(inputs=inputs, outputs=[mass_regress], name="IntNet")
    print(model.summary())

    return model


def format_qactivation(activation: str, nbits: int) -> str:
    """Format the activation function strings in a QKeras friendly way."""
    return f"quantized_{activation}({nbits}, 0)"