import os
os.environ ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, Dropout, Dense, Input, Layer
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from sklearn.utils import shuffle

def sparseFeature(feat, feat_num, embed_dim=4):
    return {'feat_name': feat, 'feat_num': feat_num, 'embed_dim': embed_dim}

def get_data():
    df = pd.read_csv("../data/covtype.csv")
    df = shuffle(df).reset_index(drop=True)

    est = KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='uniform')
    dense_features = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
                      'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
                      'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
                      'Horizontal_Distance_To_Fire_Points']
    sparse_features = ['wilderness', 'soil']
    df[dense_features] = est.fit_transform(df[dense_features])

    for feat in sparse_features:
        le = LabelEncoder()
        df[feat] = le.fit_transform(df[feat])

    features = dense_features + sparse_features
    print(features)

    feature_columns = [sparseFeature(feat, int(df[feat].max()) + 1, 8)
                       for feat in features]
    print(feature_columns)


    X = df.iloc[:,:-1].values.astype(np.int32)
    y = df.iloc[:,-1].values.astype(np.float64)

    return X, y, feature_columns

class FM(Layer):
    """
    Wide part
    """
    def __init__(self, feature_length, w_reg=1e-6):
        """
        Factorization Machine
        In DeepFM, only the first order feature and second order feature intersect are included.
        :param feature_length: A scalar. The length of features.
        :param w_reg: A scalar. The regularization coefficient of parameter w.
        """
        super(FM, self).__init__()
        self.feature_length = feature_length
        self.w_reg = w_reg

    def build(self, input_shape):
        self.w = self.add_weight(name='w', shape=(self.feature_length, 1),
                                 initializer='random_normal',
                                 regularizer=l2(self.w_reg),
                                 trainable=True)

    def call(self, inputs, **kwargs):
        """
        :param inputs: A dict with shape `(batch_size, {'sparse_inputs', 'embed_inputs'})`:
          sparse_inputs is 2D tensor with shape `(batch_size, sum(field_num))`
          embed_inputs is 3D tensor with shape `(batch_size, fields, embed_dim)`
        """
        sparse_inputs, embed_inputs = inputs['sparse_inputs'], inputs['embed_inputs']
        # first order
        first_order = tf.reduce_sum(tf.nn.embedding_lookup(self.w, sparse_inputs), axis=1)  # (batch_size, 1)
        # second order
        square_sum = tf.square(tf.reduce_sum(embed_inputs, axis=1, keepdims=True))  # (batch_size, 1, embed_dim)
        sum_square = tf.reduce_sum(tf.square(embed_inputs), axis=1, keepdims=True)  # (batch_size, 1, embed_dim)
        second_order = 0.5 * tf.reduce_sum(square_sum - sum_square, axis=2)  # (batch_size, 1)
        return first_order + second_order


class DNN(Layer):
    """
    Deep part
    """
    def __init__(self, hidden_units, activation='relu', dnn_dropout=0.):
        """
        DNN part
        :param hidden_units: A list like `[unit1, unit2,...,]`. List of hidden layer units's numbers
        :param activation: A string. Activation function.
        :param dnn_dropout: A scalar. dropout number.
        """
        super(DNN, self).__init__()
        self.dnn_network = [Dense(units=unit, activation=activation) for unit in hidden_units]
        self.dropout = Dropout(dnn_dropout)

    def call(self, inputs, **kwargs):
        x = inputs
        for dnn in self.dnn_network:
            x = dnn(x)
        x = self.dropout(x)
        return x

class DeepFM(Model):
    def __init__(self, feature_columns, hidden_units=(200, 200, 200), dnn_dropout=0.,
                 activation='relu', fm_w_reg=1e-6, embed_reg=1e-6):
        """
        DeepFM
        :param feature_columns: A list. sparse column feature information.
        :param hidden_units: A list. A list of dnn hidden units.
        :param dnn_dropout: A scalar. Dropout of dnn.
        :param activation: A string. Activation function of dnn.
        :param fm_w_reg: A scalar. The regularizer of w in fm.
        :param embed_reg: A scalar. The regularizer of embedding.
        """
        super(DeepFM, self).__init__()
        self.sparse_feature_columns = feature_columns
        self.embed_layers = {
            'embed_' + str(i): Embedding(input_dim=feat['feat_num'],
                                         input_length=1,
                                         output_dim=feat['embed_dim'],
                                         embeddings_initializer='random_normal',
                                         embeddings_regularizer=l2(embed_reg))
            for i, feat in enumerate(self.sparse_feature_columns)
        }
        self.index_mapping = []
        self.feature_length = 0
        for feat in self.sparse_feature_columns:
            self.index_mapping.append(self.feature_length)
            self.feature_length += feat['feat_num']
        self.embed_dim = self.sparse_feature_columns[0]['embed_dim']  # all sparse features have the same embed_dim
        self.fm = FM(self.feature_length, fm_w_reg)
        self.dnn = DNN(hidden_units, activation, dnn_dropout)
        self.dense = Dense(1, activation=None)

    def call(self, inputs, **kwargs):
        sparse_inputs = inputs
        # embedding
        sparse_embed = tf.concat([self.embed_layers['embed_{}'.format(i)](sparse_inputs[:, i])
                                  for i in range(sparse_inputs.shape[1])], axis=-1)  # (batch_size, embed_dim * fields)
        # wide
        sparse_inputs = sparse_inputs + tf.convert_to_tensor(self.index_mapping)
        wide_inputs = {'sparse_inputs': sparse_inputs,
                       'embed_inputs': tf.reshape(sparse_embed, shape=(-1, sparse_inputs.shape[1], self.embed_dim))}
        wide_outputs = self.fm(wide_inputs)  # (batch_size, 1)
        # deep
        deep_outputs = self.dnn(sparse_embed)
        deep_outputs = self.dense(deep_outputs)  # (batch_size, 1)
        # outputs
        outputs = tf.nn.sigmoid(tf.add(wide_outputs, deep_outputs))
        return outputs

    def summary(self):
        sparse_inputs = Input(shape=(len(self.sparse_feature_columns),), dtype=tf.int32)
        Model(inputs=sparse_inputs, outputs=self.call(sparse_inputs)).summary()

def test():
    """
    https://github.com/ZiyaoGeng/Recommender-System-with-TF2.0/tree/master/FM
    """
    X, y, feature_columns = get_data()

    dnn_dropout = 0.5
    hidden_units = [64, 32, 16]
    model = DeepFM(feature_columns=feature_columns, hidden_units=hidden_units, dnn_dropout=dnn_dropout)

    model.compile(loss="binary_crossentropy",
                  optimizer = "adam",
                  metrics = ["accuracy", keras.metrics.AUC(name = 'auc')])

    model.summary()

    history = model.fit(X, y, batch_size=245, epochs=40, validation_split=0.2)

if __name__ == '__main__':
    test()