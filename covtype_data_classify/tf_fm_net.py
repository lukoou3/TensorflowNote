import os
os.environ ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from sklearn.utils import shuffle

def sparseFeature(feat, feat_num):
    return {'feat_name': feat, 'feat_num': feat_num}

def get_data():
    df = pd.read_csv("../data/covtype.csv")
    df = shuffle(df).reset_index(drop=True)

    est = KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='uniform')
    dense_features = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
                      'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
                      'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
                      'Horizontal_Distance_To_Fire_Points']
    sparse_features = ['wilderness', 'soil']
    features = dense_features + sparse_features
    print(features)

    df[dense_features] = est.fit_transform(df[dense_features])

    for feat in sparse_features:
        le = LabelEncoder()
        df[feat] = le.fit_transform(df[feat])

    feature_columns = [sparseFeature(feat, int(df[feat].max()) + 1)
                       for feat in features]
    print(feature_columns)

    X = df.iloc[:,:-1].values.astype(np.int32)
    y = df.iloc[:,-1].values.astype(np.float64)

    return X, y, feature_columns

class FM_Layer(keras.layers.Layer):
    def __init__(self, feature_columns, k, w_reg=1e-6, v_reg=1e-6):
        """
        Factorization Machines
        :param feature_columns: A list. sparse column feature information.
        :param k: the latent vector
        :param w_reg: the regularization coefficient of parameter w
        :param v_reg: the regularization coefficient of parameter v
        """
        super(FM_Layer, self).__init__()
        self.sparse_feature_columns = feature_columns
        self.index_mapping = []
        self.feature_length = 0
        for feat in self.sparse_feature_columns:
            self.index_mapping.append(self.feature_length)
            self.feature_length += feat['feat_num']
        self.k = k
        self.w_reg = w_reg
        self.v_reg = v_reg

    def build(self, input_shape):
        self.w0 = self.add_weight(name='w0', shape=(1,),
                                  initializer=tf.zeros_initializer(),
                                  trainable=True)
        self.w = self.add_weight(name='w', shape=(self.feature_length, 1),
                                 initializer=tf.random_normal_initializer(),
                                 regularizer=l2(self.w_reg),
                                 trainable=True)
        self.V = self.add_weight(name='V', shape=(self.feature_length, self.k),
                                 initializer=tf.random_normal_initializer(),
                                 regularizer=l2(self.v_reg),
                                 trainable=True)

    def call(self, inputs, **kwargs):
        print(inputs)
        print(self.index_mapping)
        # mapping, 这一步就是为了映射idx, 为后面的embedding_lookup做准备
        inputs = inputs + tf.convert_to_tensor(self.index_mapping)
        # inputs = inputs + tf.constant(self.index_mapping)
        print(inputs)
        # first order
        first_order = self.w0 + tf.reduce_sum(tf.nn.embedding_lookup(self.w, inputs), axis=1)  # (batch_size, 1)
        # second order
        second_inputs = tf.nn.embedding_lookup(self.V, inputs)  # (batch_size, fields, embed_dim)
        square_sum = tf.square(tf.reduce_sum(second_inputs, axis=1, keepdims=True))  # (batch_size, 1, embed_dim)
        sum_square = tf.reduce_sum(tf.square(second_inputs), axis=1, keepdims=True)  # (batch_size, 1, embed_dim)
        second_order = 0.5 * tf.reduce_sum(square_sum - sum_square, axis=2)  # (batch_size, 1)
        # outputs
        outputs = first_order + second_order
        return outputs

class FM(keras.Model):
    def __init__(self, feature_columns, k, w_reg=1e-6, v_reg=1e-6):
        """
        Factorization Machines
        :param feature_columns: A list. sparse column feature information.
        :param k: the latent vector
        :param w_reg: the regularization coefficient of parameter w
        :param v_reg: the regularization coefficient of parameter v
        """
        super(FM, self).__init__()
        self.sparse_feature_columns = feature_columns
        self.fm = FM_Layer(feature_columns, k, w_reg, v_reg)

    def call(self, inputs, **kwargs):
        fm_outputs = self.fm(inputs)
        outputs = tf.nn.sigmoid(fm_outputs)
        return outputs

    def summary(self, **kwargs):
        sparse_inputs = keras.layers.Input(shape=(len(self.sparse_feature_columns),), dtype=tf.int32)
        keras.Model(inputs=sparse_inputs, outputs=self.call(sparse_inputs)).summary()

def test():
    X, y, feature_columns = get_data()

    model = FM(feature_columns=feature_columns, k=8)

    model.compile(loss="binary_crossentropy",
                  optimizer = "adam",
                  metrics = ["accuracy"])

    model.summary()

    history = model.fit(X, y, batch_size=4096, epochs=20, validation_split=0.2)

if __name__ == '__main__':
    test()