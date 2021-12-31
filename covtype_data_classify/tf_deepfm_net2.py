import os
os.environ ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Input, Dense, Embedding
from tensorflow.keras import Model
import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder

def get_data():
    df = pd.read_csv("../data/covtype.csv")
    df = shuffle(df).reset_index(drop=True)

    dense_features = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
                      'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
                      'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
                      'Horizontal_Distance_To_Fire_Points']
    sparse_features = ['wilderness', 'soil']
    features = dense_features + sparse_features

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df.iloc[:,:10] = scaler.fit_transform(df.iloc[:,:10].values)

    for feat in sparse_features:
        le = LabelEncoder()
        df[feat] = le.fit_transform(df[feat])

    feature_columns = [[denseFeature(feat) for feat in dense_features]] + \
                      [[sparseFeature(feat, int(df[feat].max()) + 1, 8) for feat in sparse_features]]

    print(feature_columns)

    X = df.iloc[:,:-1].values
    y = df.iloc[:,-1].values

    return X, y, feature_columns

def sparseFeature(feat, feat_onehot_dim, embed_dim):
    return {'feat': feat, 'feat_onehot_dim': feat_onehot_dim, 'embed_dim': embed_dim}

def denseFeature(feat):
    return {'feat': feat}

class FM_layer(Layer):
    def __init__(self, k, w_reg, v_reg):
        super().__init__()
        self.k = k
        self.w_reg = w_reg
        self.v_reg = v_reg

    def build(self, input_shape):
        self.w0 = self.add_weight(name='w0', shape=(1,),
                                  initializer=tf.zeros_initializer(),
                                  trainable=True,)
        self.w = self.add_weight(name='w', shape=(input_shape[-1], 1),
                                 initializer=tf.random_normal_initializer(),
                                 trainable=True,
                                 regularizer=tf.keras.regularizers.l2(self.w_reg))
        self.v = self.add_weight(name='v', shape=(input_shape[-1], self.k),
                                 initializer=tf.random_normal_initializer(),
                                 trainable=True,
                                 regularizer=tf.keras.regularizers.l2(self.v_reg))

    def call(self, inputs, **kwargs):
        linear_part = tf.matmul(inputs, self.w) + self.w0   #shape:(batchsize, 1)

        inter_part1 = tf.pow(tf.matmul(inputs, self.v), 2)  #shape:(batchsize, self.k)
        inter_part2 = tf.matmul(tf.pow(inputs, 2), tf.pow(self.v, 2)) #shape:(batchsize, self.k)
        inter_part = 0.5*tf.reduce_sum(inter_part1 - inter_part2, axis=-1, keepdims=True) #shape:(batchsize, 1)

        output = linear_part + inter_part
        return output

class Dense_layer(Layer):
    def __init__(self, hidden_units, output_dim, activation):
        super().__init__()
        self.hidden_units = hidden_units
        self.output_dim = output_dim
        self.activation = activation

        self.hidden_layer = [Dense(i, activation=self.activation)
                             for i in self.hidden_units]
        self.output_layer = Dense(self.output_dim, activation=None)

    def call(self, inputs):
        x = inputs
        for layer in self.hidden_layer:
            x = layer(x)
        output = self.output_layer(x)
        return output

class DeepFM(Model):
    def __init__(self, feature_columns, k, w_reg, v_reg, hidden_units, output_dim, activation):
        super().__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.embed_layers = {
            'embed_' + str(i): Embedding(feat['feat_onehot_dim'], feat['embed_dim'])
            for i, feat in enumerate(self.sparse_feature_columns)
        }

        self.FM = FM_layer(k, w_reg, v_reg)
        self.Dense = Dense_layer(hidden_units, output_dim, activation)

    def call(self, inputs):
        dense_inputs, sparse_inputs = inputs[:, :10], inputs[:, 10:]
        # embedding
        sparse_embed = tf.concat([self.embed_layers['embed_{}'.format(i)](sparse_inputs[:, i])
                                  for i in range(sparse_inputs.shape[1])], axis=1)
        x = tf.concat([dense_inputs, sparse_embed], axis=-1)

        fm_output = self.FM(x)
        dense_output = self.Dense(x)
        output = tf.nn.sigmoid(0.5*(fm_output + dense_output))
        return output

    def summary(self):
        inputs = Input(shape=(12,), dtype=tf.float32)
        Model(inputs=inputs, outputs=self.call(inputs)).summary()

def test():
    """
    https://zhuanlan.zhihu.com/p/361451464
    https://github.com/jc-LeeHub/Recommend-System-tf2.0/tree/master/DeepFM
    :return:
    """
    X, y, feature_columns = get_data()
    k = 10
    w_reg = 1e-4
    v_reg = 1e-4
    hidden_units = [64, 32, 16]
    output_dim = 1
    activation = 'relu'

    model = DeepFM(feature_columns, k, w_reg, v_reg, hidden_units, output_dim, activation)

    model.compile(loss="binary_crossentropy",
                  optimizer = "adam",
                  metrics = ["accuracy", keras.metrics.AUC(name = 'auc')])

    model.summary()

    history = model.fit(X, y, batch_size=4096, epochs=100, validation_split=0.2)

if __name__ == '__main__':
    test()