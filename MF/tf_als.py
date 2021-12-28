import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.regularizers import l2
from sklearn.utils import shuffle

def get_data():
    #df = pd.read_csv("../data/ua.base", sep="\t", header=None, names=["user_id", "movie_id", "rating", "ts"])
    df = pd.read_csv("../data/ml-1m/ratings.dat", sep="::", header=None, names=["user_id", "movie_id", "rating", "ts"])
    #df = pd.read_csv(r"D:\kaggle-data\ml-1m\ratings.dat", sep="::", header=None, names=["user_id", "movie_id", "rating", "ts"])
    df = shuffle(df).reset_index(drop=True)
    print(df.dtypes)

    id_cols = ["user_id", "movie_id"]
    id_map = {}
    size = 0

    for col in id_cols:
        id_map[col] = {v: i for i, v in enumerate(df[col].unique())}
        size += len(id_map[col])
    for col in id_cols:
        df[col] = df[col].map(id_map[col])

    id_sizes = {col: len(map) for col, map in id_map.items()}
    print(size)
    print(id_sizes)
    print(df.dtypes)

    x = df[["user_id", "movie_id"]].values.astype('int32')
    y = df["rating"].values

    print({'x_dtype': x.dtype, 'y_dtype': y.dtype})

    return x, y


def test_als_func_api():
    x, y = get_data()
    x_u = x[:, [0]]
    x_i = x[:, [1]]

    usize = x_u.max() + 1
    isize = x_i.max() + 1
    k = 16

    input_u = keras.layers.Input(shape=[1])
    input_i = keras.layers.Input(shape=[1])

    embedding_u = keras.layers.Embedding(usize, k, embeddings_regularizer=l2(1e-4))(input_u) # [None, fields_length, embed_dim]
    embedding_i = keras.layers.Embedding(isize, k, embeddings_regularizer=l2(1e-4))(input_i) # [None, fields_length, embed_dim]
    # target_shape: Target shape. Tuple of integers, does not include the samples dimension (batch size).
    embedding_u = keras.layers.Reshape([k])(embedding_u) # [None, embed_dim]
    embedding_i = keras.layers.Reshape([-1])(embedding_i) # [None, embed_dim]
    output = keras.layers.Dot(1)([embedding_u, embedding_i])

    model = keras.models.Model(inputs=[input_u, input_i], outputs=[output])

    model.compile(
        loss=keras.losses.mean_squared_error,
        # optimizer=keras.optimizers.SGD(),
        optimizer=keras.optimizers.Adam(),
        metrics=["mse"]
    )
    model.summary()

    history = model.fit([x_u, x_i], y, batch_size=256, epochs=10, validation_split=0.1)

def test_als_func_api_lambda():
    x, y = get_data()
    x_u = x[:, [0]]
    x_i = x[:, [1]]

    usize = x_u.max() + 1
    isize = x_i.max() + 1
    k = 16

    input_u = keras.layers.Input(shape=[1])
    input_i = keras.layers.Input(shape=[1])

    embedding_u = keras.layers.Embedding(usize, k, embeddings_regularizer=l2(1e-4))(input_u) # [None, fields_length, embed_dim]
    embedding_i = keras.layers.Embedding(isize, k, embeddings_regularizer=l2(1e-4))(input_i) # [None, fields_length, embed_dim]
    # target_shape: Target shape. Tuple of integers, does not include the samples dimension (batch size).
    embedding_u = keras.layers.Reshape([k])(embedding_u) # [None, embed_dim]
    embedding_i = keras.layers.Reshape([-1])(embedding_i) # [None, embed_dim]
    if True:
        multiply = keras.layers.multiply([embedding_u, embedding_i])
        output = keras.layers.Lambda(lambda i: tf.reduce_sum(i, axis=-1))(multiply)
    else:
        output = keras.layers.Lambda(lambda inputs: tf.reduce_sum(inputs[0] * inputs[1], axis=-1))([embedding_u, embedding_i])

    model = keras.models.Model(inputs=[input_u, input_i], outputs=[output])

    model.compile(
        loss=keras.losses.mean_squared_error,
        # optimizer=keras.optimizers.SGD(),
        optimizer=keras.optimizers.Adam(),
        metrics=["mse"]
    )
    model.summary()

    history = model.fit([x_u, x_i], y, batch_size=256, epochs=10, validation_split=0.1)

class AlsLayer(keras.layers.Layer):
    def __init__(self, user_num, item_num, latent_dim=20):
        super().__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.latent_dim = latent_dim

    def build(self, input_shape):
        """构建所需要的参数"""
        # [TensorShape([None, 1]), TensorShape([None, 1])]
        print(input_shape)
        self.u = self.add_weight(name='u',
                                 shape=(self.user_num, self.latent_dim),
                                 initializer='random_normal',
                                 trainable=True)
        self.i = self.add_weight(name='i',
                                 shape=(self.item_num, self.latent_dim),
                                 initializer='random_normal',
                                 trainable=True)

    def call(self, inputs):
        """完成正向计算"""
        user_id, item_id = inputs
        latent_user = tf.nn.embedding_lookup(params=self.u, ids=user_id) # (None, 1, k)
        latent_item = tf.nn.embedding_lookup(params=self.i, ids=item_id) # (None, 1, k)
        outputs = tf.reduce_sum(tf.multiply(latent_user, latent_item), axis=-1)  # (None, 1)
        print(latent_user)
        print(latent_item)
        print(outputs)
        return outputs

def test_als_func_api_layer():
    x, y = get_data()
    x_u = x[:, [0]]
    x_i = x[:, [1]]

    usize = x_u.max() + 1
    isize = x_i.max() + 1
    k = 16

    # 这里还必须声明dtype为int32, layers.Embedding不用转是因为它的call函数中会判断类型不是int会转换: inputs = math_ops.cast(inputs, 'int32')
    input_u = keras.layers.Input(shape=[1], dtype='int32')
    input_i = keras.layers.Input(shape=[1], dtype='int32')

    output = AlsLayer(usize, isize, k)([input_u, input_i])
    model = keras.models.Model(inputs=[input_u, input_i], outputs=[output])

    model.compile(
        loss=keras.losses.mean_squared_error,
        # optimizer=keras.optimizers.SGD(),
        optimizer=keras.optimizers.Adam(),
        metrics=["mse"]
    )
    model.summary()

    #history = model.fit([x_u, x_i], y, batch_size=256, epochs=10, validation_split=0.1)

class AlsWithBiasLayer(keras.layers.Layer):
    def __init__(self, user_num, item_num, latent_dim=20):
        super().__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.latent_dim = latent_dim

    def build(self, input_shape):
        """构建所需要的参数"""
        # [TensorShape([None, 1]), TensorShape([None, 1])]
        print(input_shape)
        self.u = self.add_weight(name='user_latent_matrix',
                                 shape=(self.user_num, self.latent_dim),
                                 initializer='random_normal',
                                 trainable=True)
        self.i = self.add_weight(name='item_latent_matrix',
                                 shape=(self.item_num, self.latent_dim),
                                 initializer='random_normal',
                                 trainable=True)
        self.u_bias = self.add_weight(name='user_bias',
                                    shape=(self.user_num,),
                                    initializer='zeros',
                                    trainable=True)
        self.i_bias = self.add_weight(name='item_bias',
                                      shape=(self.item_num,),
                                      initializer='zeros',
                                      trainable=True)

    def call(self, inputs):
        """完成正向计算"""
        user_id, item_id = inputs
        # 配置shape=()时, 输出就是(None, k)
        # 同样add_weight中维度省略1时输出也会少一个1的维度
        latent_user = tf.nn.embedding_lookup(params=self.u, ids=user_id) # (None, 1, k)
        latent_item = tf.nn.embedding_lookup(params=self.i, ids=item_id) # (None, 1, k)
        user_bias = tf.nn.embedding_lookup(params=self.u_bias, ids=user_id) # (None, 1)
        item_bias = tf.nn.embedding_lookup(params=self.i_bias, ids=item_id) # (None, 1)
        outputs = tf.reduce_sum(tf.multiply(latent_user, latent_item), axis=-1)  # (None, 1)
        outputs = outputs + user_bias + item_bias
        return outputs

def test_als_func_api_layer_bias():
    x, y = get_data()
    x_u = x[:, [0]]
    x_i = x[:, [1]]

    usize = x_u.max() + 1
    isize = x_i.max() + 1
    k = 16

    # 这里还必须声明dtype为int32, layers.Embedding不用转是因为它的call函数中会判断类型不是int会转换: inputs = math_ops.cast(inputs, 'int32')
    input_u = keras.layers.Input(shape=[1], dtype='int32')
    input_i = keras.layers.Input(shape=[1], dtype='int32')

    output = AlsWithBiasLayer(usize, isize, k)([input_u, input_i])
    model = keras.models.Model(inputs=[input_u, input_i], outputs=[output])

    model.compile(
        loss=keras.losses.mean_squared_error,
        # optimizer=keras.optimizers.SGD(),
        optimizer=keras.optimizers.Adam(),
        metrics=["mse"]
    )
    model.summary()

    history = model.fit([x_u, x_i], y, batch_size=256, epochs=20, validation_split=0.1)

if __name__ == '__main__':
    # test_als_func_api()
    # test_als_func_api_lambda()
    # test_als_func_api_layer()
    test_als_func_api_layer_bias()
