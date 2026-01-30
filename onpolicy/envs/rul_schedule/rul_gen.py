import pandas as pd
import numpy as np
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow import reshape
import warnings
import pickle
from pathlib import Path
import os
import time
from typing import Optional
# %matplotlib inline
import tensorflow as tf
from scipy.sparse.linalg import eigs
from tensorflow.keras import layers
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import random
import pickle

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        # Optionally cap memory if desired; keep safe defaults across devices.
        # If this fails (e.g., driver restrictions), continue without hard limits.
    except Exception:
        pass

# GCPATr Definition and Helper funtions

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_DATA_DICT = _REPO_ROOT / 'models' / 'model' / 'CMAPSS_data_dictionary.pkl'
with open(str(_DEFAULT_DATA_DICT), 'rb') as f:
    data_dictionary = pickle.load(f)

cov_adj_mat = data_dictionary['FD001']['Cov_mat']

def masked_mape_np(preds, labels, null_val=np.nan):
  with np.errstate(divide='ignore', invalid='ignore'):
    if np.isnan(null_val):
      mask = ~np.isnan(labels)
    else:
      mask = np.not_equal(labels, null_val)
      mask = mask.astype('float32')
      mask /= np.mean(mask)
      mape = np.abs(np.divide(np.subtract(preds, labels).astype('float32'), labels))
      mape = np.nan_to_num(mask * mape)
  return np.mean(mape)

def score_metric(y_true, y_pred):
    c = y_pred - y_true
    a = tf.math.exp(-c/13) - 1
    b = tf.math.exp(c/10) - 1
    return tf.reduce_sum(tf.where(c < 0, a, b))

# Learning Rate Scheduler
def scheduler(epoch, lr):
    exp = np.floor((1 + epoch) / 100)
    alpha = 0.00018 * (0.75 ** exp)
    return float(alpha)

class SpatialConvLayer(tf.keras.layers.Layer):
    def __init__(self, Ks, c_in, c_out, kernel, **kwargs):
        super(SpatialConvLayer, self).__init__(**kwargs)
        self.Ks = Ks
        self.c_in = c_in
        self.c_out = c_out
        self.kernel = kernel

        self.w_input = self.add_weight(shape=(1, 1, c_in, c_out), initializer="random_normal", dtype="float32", trainable=True)

        self.ws = self.add_weight(shape=(Ks * c_in, c_out), initializer="random_normal", dtype="float32", trainable=True)

        self.bs = self.add_weight(shape=(c_out,), initializer="random_normal", dtype="float32", trainable=True)

    # def build(self, input_shape):

    def call(self, inputs):
        _, T, n, _ = inputs.get_shape().as_list()
        if self.c_in > self.c_out:
            # bottleneck down-sampling
            x_input = tf.nn.conv2d(inputs, self.w_input, strides=[1, 1, 1, 1], padding='SAME')
        elif self.c_in < self.c_out:
            # if the size of input channel is less than the output,
            # padding x to the same size of output channel.
            # Note, _.get_shape() cannot convert a partially known TensorShape to a Tensor.
            x_input = tf.concat([inputs, tf.zeros([tf.shape(inputs)[0], T, n, self.c_out - self.c_in])], axis=3)
        else:
            x_input = inputs

        # x_gconv = self.gconv(tf.reshape(inputs, [-1, n, self.c_in]), self.ws) + self.bs

        n = tf.shape(self.kernel)[0]
        x_tmp = tf.reshape(tf.transpose(tf.reshape(inputs, [-1, n, self.c_in]), [0, 2, 1]), [-1, n])
        x_mul = tf.reshape(tf.matmul(x_tmp, self.kernel), [-1, self.c_in, self.Ks, n])
        x_ker = tf.reshape(tf.transpose(x_mul, [0, 3, 1, 2]), [-1, self.c_in * self.Ks])
        x_gconv = tf.reshape(tf.matmul(x_ker, self.ws), [-1, n, self.c_out]) + self.bs

        x_gc = tf.reshape(x_gconv, [-1, T, n, self.c_out])
        return tf.nn.relu(x_gc[:, :, :, 0:self.c_out] + x_input)


def scaled_laplacian(W):
    '''
    Normalized graph Laplacian function.
    :param W: np.ndarray, [n_route, n_route], weighted adjacency matrix of G.
    :return: np.matrix, [n_route, n_route].
    '''
    # d ->  diagonal degree matrix
    n, d = np.shape(W)[0], np.sum(W, axis=1)
    # L -> graph Laplacian
    L = -W
    L[np.diag_indices_from(L)] = d
    for i in range(n):
        for j in range(n):
            if (d[i] > 0) and (d[j] > 0):
                L[i, j] = L[i, j] / np.sqrt(d[i] * d[j])
    # lambda_max \approx 2.0, the largest eigenvalues of L.
    lambda_max = eigs(L, k=1, which='LR')[0][0].real
    return np.mat(2 * L / lambda_max - np.identity(n))


def cheb_poly_approx(L, Ks, n):
    '''
    Chebyshev polynomials approximation function.
    :param L: np.matrix, [n_route, n_route], graph Laplacian.
    :param Ks: int, kernel size of spatial convolution.
    :param n: int, number of routes / size of graph.
    :return: np.ndarray, [n_route, Ks*n_route].
    '''
    L0, L1 = np.mat(np.identity(n)), np.mat(np.copy(L))

    if Ks > 1:
        L_list = [np.copy(L0), np.copy(L1)]
        for i in range(Ks - 2):
            Ln = np.mat(2 * L * L1 - L0)
            L_list.append(np.copy(Ln))
            L0, L1 = np.matrix(np.copy(L1)), np.matrix(np.copy(Ln))
        # L_lsit [Ks, n*n], Lk [n, Ks*n]
        return np.concatenate(L_list, axis=-1)
    elif Ks == 1:
        return np.asarray(L0)
    else:
        raise ValueError(f'ERROR: the size of spatial kernel must be greater than 1, but received "{Ks}".')


def first_approx(W, n):
    '''
    1st-order approximation function.
    :param W: np.ndarray, [n_route, n_route], weighted adjacency matrix of G.
    :param n: int, number of routes / size of graph.
    :return: np.ndarray, [n_route, n_route].
    '''
    A = W + np.identity(n)
    d = np.sum(A, axis=1)
    sinvD = np.sqrt(np.mat(np.diag(d)).I)
    # refer to Eq.5
    return np.mat(np.identity(n) + sinvD * A * sinvD)


def weight_matrix(file_path, sigma2=0.1, epsilon=0.5, scaling=True):
    '''
    Load weight matrix function.
    :param file_path: str, the path of saved weight matrix file.
    :param sigma2: float, scalar of matrix W.
    :param epsilon: float, thresholds to control the sparsity of matrix W.
    :param scaling: bool, whether applies numerical scaling on W.
    :return: np.ndarray, [n_route, n_route].
    '''
    #try:
        #W = pd.read_csv(file_path, header=None).values
        # W = W.astype('float64')
    #except FileNotFoundError:
    #    print(f'ERROR: input file was not found in {file_path}.')

    #W = 100 * np.random.rand(25, 25)
    W = cov_adj_mat

    # check whether W is a 0/1 matrix.
    if set(np.unique(W)) == {0, 1}:
        print('The input graph is a 0/1 matrix; set "scaling" to False.')
        scaling = False

    if scaling:
        n = W.shape[0]
        W = W / 10000.
        W2, W_mask = W * W, np.ones([n, n]) - np.identity(n)
        # refer to Eq.10
        return np.exp(-W2 / sigma2) * (np.exp(-W2 / sigma2) >= epsilon) * W_mask
    else:
        return W

adj_mat_path = "Hello"

class PositionEmbeddingLayer(layers.Layer):
    def __init__(self, sequence_length, output_dim, **kwargs):
        super(PositionEmbeddingLayer, self).__init__(**kwargs)
        self.position_embedding_layer = layers.Embedding(
            input_dim=(sequence_length), output_dim=output_dim
        )
        self.sequence_length = sequence_length

    def call(self, inputs):
        position_indices = tf.range(self.sequence_length)  #tf.range(1, self.sequence_length + 1, 1)
        embedded_words = inputs
        embedded_indices = self.position_embedding_layer(position_indices)
        return embedded_words + embedded_indices


#GCPATr Model
@tf.keras.utils.register_keras_serializable()
class GCPATr(keras.Model):
    def __init__(self, num_features, seq_len, num_attn_heads, hidden_layer_dim, num_transformer_blocks, **kwargs):
        super().__init__(**kwargs)

        # Calculate graph kernel
        W = weight_matrix('hello')
        Ks = 3  # Ks: int, kernel size of spatial convolution.
        L = scaled_laplacian(W)
        Lk = cheb_poly_approx(L, Ks, num_features)
        graph_kernel = tf.cast(tf.constant(Lk), tf.float32)
        self.num_features = num_features
        self.num_heads = num_attn_heads
        self.seq_len = seq_len
        self.hidden_layer_dim = hidden_layer_dim
        self.num_transformer_blocks = num_transformer_blocks
        self.rate = 0.1

        self.blocks = []
        for i in range(num_transformer_blocks):
            WqL = []
            WkL = []
            WvL = []
            tq = []

            scl = SpatialConvLayer(Ks, 1, 1, graph_kernel)
            el = layers.Dense(num_features)
            layernorm1 = layers.LayerNormalization(epsilon=1e-6)
            layernorm2 = layers.LayerNormalization(epsilon=1e-6)
            dropout1 = layers.Dropout(self.rate)
            dropout2 = layers.Dropout(self.rate)

            for i in range(self.num_heads):
                WqL.append(SpatialConvLayer(Ks, 1, 1, graph_kernel))
                WkL.append(SpatialConvLayer(Ks, 1, 1, graph_kernel))
                WvL.append(SpatialConvLayer(Ks, 1, 1, graph_kernel))
                tq.append(keras.Sequential(
                [layers.Dense(64, activation="relu"), layers.Dense(64, activation="relu"), layers.Dense(2, activation="softmax"),]))

            Wlt = self.add_weight(shape=((self.num_heads * int(num_features)), int(num_features)), initializer="random_normal", trainable=True)

            #block component dictionary
            block_dict = {"WqL":WqL,
                          "WkL":WkL,
                          "WvL":WvL,
                          "tq":tq,
                          "scl":scl,
                          "el":el,
                          "layernorm1":layernorm1,
                          "layernorm2":layernorm2,
                          "dropout1":dropout1,
                          "dropout2":dropout2,
                          "Wlt":Wlt
                        }
            self.blocks.append(block_dict)

        self.embedding_layer = PositionEmbeddingLayer(seq_len, num_features)
        self.global_average_pooling = layers.GlobalAveragePooling1D()
        self.dropout1 = layers.Dropout(0.2)
        self.dropout2 = layers.Dropout(0.2)
        self.linear1 = layers.Dense(32, activation="relu")
        self.linear2 = layers.Dense(1)

    def get_config(self):  # Add get_config method
        config = super().get_config()
        config.update({
            "num_features": self.num_features,
            "num_attn_heads": self.num_heads,
            "seq_len": self.seq_len,
            "hidden_layer_dim": self.hidden_layer_dim,
            "num_transformer_blocks": self.num_transformer_blocks
        })
        return config

    @classmethod
    def from_config(cls, config):  # Add from_config method
        return cls(**config)

    def call(self, inputs, training=True):

        x = self.embedding_layer(inputs)

        for blk in self.blocks:

            # Adding additional dimension for compatibility
            x_tran = tf.expand_dims(x, axis=3)
            a_xL = []

            # Generate Query, Key and Value corresponding to each attention head
            for i in range(self.num_heads):

                # Query : batch_size x time_steps x dq
                xq = blk['WqL'][i](x_tran)
                xq = tf.squeeze(xq, [3])

                # Key : batch_size x time_steps x dk
                xk = blk['WkL'][i](x_tran)
                xk = tf.squeeze(xk, [3])

                # Value : batch_size x time_steps x dv
                xv = blk['WvL'][i](x_tran)
                xv = tf.squeeze(xv, [3])

                # Transposing each key in a batch (xk_t : batch_size x dk x time_steps)
                xk_t = tf.transpose(xk, perm=[0, 2, 1])

                # Computing scaled dot product self attention of each time step in each training sample (s_a : batch_size x time_steps x time_steps)
                s_a = tf.math.multiply(tf.keras.layers.Dot(axes=(1, 2))([xk_t, xq]), (1/self.num_features))

                # Applying Softmax Layer to the self attention weights for proper scaling (sft_s_a : batch_size x time_steps x time_steps)
                sft_s_a = tf.keras.layers.Softmax(axis=2)(s_a)

                # Temporal Information
                tmul_a = blk['tq'][i](tf.transpose(sft_s_a, perm=[0, 2, 1]))

                # Computing attention augmented values for each time step and each training sample (a_x : batch_size x time_steps x dim)
                auv = tf.keras.layers.Dot(axes=(1, 2))([xv, sft_s_a])
                auv = tf.transpose(auv, perm=[0, 2, 1])

                # Temporal Information Fusion
                r = tf.concat([tf.expand_dims(auv, 3), tf.expand_dims(xv, 3)], 3)
                g = tf.expand_dims(tmul_a, 2)
                a_xL.append(tf.math.reduce_sum(tf.math.multiply(r, g), axis=3))

            # Concatenate and applying linear transform for making dimensions compatible
            a_x = tf.concat(a_xL, -1)
            a_x_tran = tf.matmul(a_x, blk['Wlt'])

            a_x_tran = blk['dropout1'](a_x_tran, training=training)
            out1 = blk['layernorm1'](x + a_x_tran)
            ffn_output = tf.expand_dims(out1, axis=3)
            ffn_output = blk['scl'](ffn_output)
            ffn_output = tf.squeeze(ffn_output, [3])
            ffn_output = blk['el'](ffn_output)
            ffn_output = blk['dropout2'](ffn_output, training=training)
            x = blk['layernorm2'](out1 + ffn_output)

        x = self.global_average_pooling(x)
        x = self.dropout1(x, training=training)
        x = self.linear1(x)
        x = self.dropout2(x, training=training)
        return self.linear2(x)
    
class rul_prediction():
  def __init__(self, model_path, lookback_window, custom_objects=None):

    # custom_objects is the dictionary of all custom layers in the model
    # Load the RUL prediction model
    self.model = tf.keras.models.load_model(
                 model_path,
                 custom_objects=custom_objects
                )

    # Lookback window size for the prediction models input
    self.lw = lookback_window

    # Initialize Observation list storing per step observation (max size lw)
    self.ol = []

  def add_observation(self, obs):
    # Add new observation to observation list
    # If list has lookback_window number of elements then remove first entry and add new entry yp last
    # make sure obs has dimensions (1, number_of_inputs)
    if len(obs.shape) == 1:
      obs = np.expand_dims(obs, axis=0)

    if len(self.ol) == self.lw:
      self.ol.pop(0)
      self.ol.append(obs)
    else:
      self.ol.append(obs)

  def predict_rul(self):
    # Predict RUL using the observations in the observation list
    # If there are fewer than lookback window elements return RUL as 125
    if len(self.ol) < self.lw:
      return 125.0

    # convert observations in the observation list into one input matrix
    model_imput = np.expand_dims(np.concatenate(self.ol, axis=0), axis=0)

    # Return RUL prediction from model
    return self.model(model_imput, training=False).numpy()[0, 0]

  def reset(self):
    # Reset the observation list
    self.ol = []

  def add_obs_and_predict_rul(self, obs):
    # Add observation and predict RUL
    self.add_observation(obs)
    return self.predict_rul()


class predictor():
    # Load the model from the file
    def __init__(self, lookback_window=40, model_path: Optional[str] = None):
        if model_path is None:
            model_path = str(_REPO_ROOT / 'models' / 'model')
        self.lw = lookback_window
        custom_objects={"GCPATr": GCPATr,
                        "PositionEmbeddingLayer": PositionEmbeddingLayer,
                        "SpatialConvLayer": SpatialConvLayer
                        }
        self.model = tf.keras.models.load_model(
                model_path,
                custom_objects=custom_objects
                )
    
    def predict(self, obs):
        if len(obs) < self.lw:
            return 125.0
        
        model_input = np.expand_dims(np.concatenate(obs, axis=0), axis=0)
        return self.model(model_input, training=False).numpy()[0, 0]