import os
os.environ ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import pandas as pd


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.regularizers import l2
from sklearn.utils import shuffle

tf.debugging.set_log_device_placement (True)     # 设置输出运算所在的设备

cpus = tf.config.list_physical_devices ('CPU')   # 获取当前设备的 CPU 列表
tf.config.set_visible_devices (cpus)             # 设置 TensorFlow 的可见设备范围为 cpu


def not_process_cate_type():
    array = np.loadtxt("data/covtype_scaler.csv", delimiter=",", skiprows=1)
    # array = np.loadtxt("data/covtype.csv", delimiter=",", skiprows=1, max_rows=None)
    array = shuffle(array)
    print(array.min(axis=0))
    print(array.max(axis=0))

    print(array[:10])

    x = array[:, :-1]
    y = array[:, -1]

    input = keras.layers.Input(shape=x.shape[1:])
    hidden = keras.layers.Dense(1)(input)
    output = keras.layers.Lambda(lambda x: tf.nn.sigmoid(x))(hidden)
    model = keras.models.Model(inputs=[input], outputs=[output])

    model.compile(
        loss=keras.losses.binary_crossentropy,
        optimizer=keras.optimizers.SGD(),
        metrics=["accuracy"]
    )
    model.summary()

    history = model.fit(x, y, batch_size=128, epochs=10, validation_split=0.2)


def process_cate_type():
    df = pd.read_csv("data/covtype_scaler.csv")
    df = shuffle(df).reset_index(drop=True)
    cate_cols = ["wilderness", "soil"]
    cate_map = {}

    size = 0
    for col in cate_cols:
        cate_map[col] = {v: i for i, v in enumerate(df[col].unique(), start=size)}
        size += len(cate_map[col])
    for col in cate_cols:
        df[col] = df[col].map(cate_map[col])

    cate_size = sum([len(map) for col, map in cate_map.items()])
    print(cate_map)
    print(cate_size)

    array = df.values
    x_dense = array[:, :-3]
    x_spase = array[:, -3:-1]
    y = array[:, -1]

    input_dense = keras.layers.Input(shape=x_dense.shape[1:])
    weight_dense = keras.layers.Dense(1, kernel_regularizer='l2', bias_regularizer='l2')(input_dense)
    input_spase = keras.layers.Input(shape=x_spase.shape[1:])
    embedding_spase = keras.layers.Embedding(cate_size, 1, input_length=2, embeddings_regularizer='l2')(input_spase)
    weight_spase = keras.layers.Reshape((2,))(embedding_spase)
    weight = keras.layers.concatenate([weight_dense, weight_spase])
    output = keras.layers.Lambda(lambda x: tf.nn.sigmoid(tf.reduce_sum(x, axis=-1, keepdims=True)))(weight)

    model = keras.models.Model(inputs=[input_dense, input_spase], outputs=[output])

    model.compile(
        loss=keras.losses.binary_crossentropy,
        optimizer=keras.optimizers.SGD(),
        metrics=["accuracy"]
    )
    model.summary()

    history = model.fit([x_dense, x_spase], y, batch_size=128, epochs=20, validation_split=0.2)


def process_cate_type_dnn():
    df = pd.read_csv("data/covtype_scaler.csv")
    df = shuffle(df).reset_index(drop=True)
    cate_cols = ["wilderness", "soil"]
    cate_map = {}

    size = 0
    for col in cate_cols:
        cate_map[col] = {v: i for i, v in enumerate(df[col].unique(), start=size)}
        size += len(cate_map[col])
    for col in cate_cols:
        df[col] = df[col].map(cate_map[col])

    cate_size = sum([len(map) for col, map in cate_map.items()])
    print(cate_map)
    print(cate_size)
    print(size)

    array = df.values
    x_dense = array[:, :-3]
    x_spase = array[:, -3:-1]
    y = array[:, -1]

    input_dense = keras.layers.Input(shape=x_dense.shape[1:])

    input_spase = keras.layers.Input(shape=x_spase.shape[1:])
    embedding_spase = keras.layers.Embedding(cate_size, 10, input_length=2)(input_spase)
    weight_spase = keras.layers.Reshape((-1,))(embedding_spase)
    weight = keras.layers.concatenate([input_dense, weight_spase])
    output = keras.layers.Dense(64, activation='relu')(weight)
    output = keras.layers.Dense(32, activation='relu')(weight)
    output = keras.layers.Dense(16, activation='relu')(output)
    output = keras.layers.Dense(1, activation="sigmoid")(output)

    model = keras.models.Model(inputs=[input_dense, input_spase], outputs=[output])

    model.compile(
        loss=keras.losses.binary_crossentropy,
        optimizer=keras.optimizers.SGD(),
        metrics=["accuracy"]
    )
    model.summary()

    history = model.fit([x_dense, x_spase], y, batch_size=128, epochs=10, validation_split=0.2)


def process_cate_type_dnn2():
    df = pd.read_csv("data/covtype_scaler.csv")
    df = shuffle(df).reset_index(drop=True)
    cate_cols = ["wilderness", "soil"]
    cate_map = {}

    size = 0
    for col in cate_cols:
        cate_map[col] = {v: i for i, v in enumerate(df[col].unique())}
        size += len(cate_map[col])
    for col in cate_cols:
        df[col] = df[col].map(cate_map[col])

    cate_size = sum([len(map) for col, map in cate_map.items()])
    print(cate_map)
    print(cate_size)

    array = df.values
    x_dense = array[:, :-3]
    x_spase1 = array[:, [-3]]
    x_spase2 = array[:, [-2]]
    y = array[:, -1]

    input_dense = keras.layers.Input(shape=x_dense.shape[1:])

    input_spase1 = keras.layers.Input(shape=x_spase1.shape[1:])
    input_spase2 = keras.layers.Input(shape=x_spase2.shape[1:])

    embedding_spase1 = keras.layers.Embedding(len(cate_map["wilderness"]), 1)(input_spase1)
    embedding_spase2 = keras.layers.Embedding(len(cate_map["soil"]), 4)(input_spase2)

    output = keras.layers.concatenate([
        input_dense,
        keras.layers.Flatten()(embedding_spase1),
        keras.layers.Flatten()(embedding_spase2)
    ])
    output = keras.layers.Dense(128)(output)
    output = keras.layers.Dense(64)(output)
    output = keras.layers.Dense(32)(output)
    output = keras.layers.Dense(16)(output)
    output = keras.layers.Dense(1, activation="sigmoid")(output)

    model = keras.models.Model(inputs=[input_dense, input_spase1, input_spase2], outputs=[output])

    model.compile(
        loss=keras.losses.binary_crossentropy,
        optimizer=keras.optimizers.SGD(),
        metrics=["accuracy"]
    )
    model.summary()

    history = model.fit([x_dense, x_spase1, x_spase2], y, batch_size=128, epochs=20, validation_split=0.2)


def process_cate_type_dnn3():
    df = pd.read_csv("data/covtype_scaler.csv")
    df = shuffle(df).reset_index(drop=True)
    cate_cols = ["wilderness", "soil"]
    cate_map = {}

    size = 0
    for col in cate_cols:
        cate_map[col] = {v: i for i, v in enumerate(df[col].unique(), start=size)}
        size += len(cate_map[col])
    for col in cate_cols:
        df[col] = df[col].map(cate_map[col])

    cate_size = sum([len(map) for col, map in cate_map.items()])
    print(cate_map)
    print(cate_size)

    array = df.values
    x_dense = array[:, :-3]
    x_spase = array[:, -3:-1]
    y = array[:, -1]

    embed_reg = 1e-6
    dense_reg = 1e-6

    input_dense = keras.layers.Input(shape=x_dense.shape[1:])

    input_spase = keras.layers.Input(shape=x_spase.shape[1:])
    embedding_spase = keras.layers.Embedding(cate_size, 10, embeddings_initializer='random_normal',
                                             embeddings_regularizer=l2(embed_reg))(input_spase)
    weight_spase = keras.layers.Flatten()(embedding_spase)
    output = keras.layers.concatenate([input_dense, weight_spase])
    output = keras.layers.Dense(64, activation='relu')(output)
    output = keras.layers.Dense(32, activation='relu')(output)
    output = keras.layers.Dense(16, activation='relu')(output)
    output = keras.layers.Dense(1, activation="sigmoid")(output)

    model = keras.models.Model(inputs=[input_dense, input_spase], outputs=[output])

    model.compile(
        loss=keras.losses.binary_crossentropy,
        optimizer=keras.optimizers.SGD(),
        metrics=["accuracy"]
    )
    model.summary()

    history = model.fit([x_dense, x_spase], y, batch_size=128, epochs=20, validation_split=0.2)


def process_cate_type_dnn4():
    df = pd.read_csv("data/covtype.csv", dtype="float32")
    df = shuffle(df).reset_index(drop=True)
    cate_cols = ["wilderness", "soil"]

    from sklearn.preprocessing import OneHotEncoder
    onehot_encoder = OneHotEncoder(sparse=False)
    onehot_encoded = onehot_encoder.fit_transform(df[cate_cols])

    from sklearn.preprocessing import StandardScaler

    array = df.values
    scaler = StandardScaler()
    array[:, :10] = scaler.fit_transform(array[:, :10])

    x = np.concatenate([array[:, :-3], onehot_encoded], axis=1)
    y = array[:, -1]

    print(x.shape)
    print(y.shape)

    embed_reg = 0

    input = keras.layers.Input(shape=x.shape[1:])
    output = keras.layers.Dense(64, activation='relu')(input)
    output = keras.layers.Dense(32, activation='relu')(output)
    output = keras.layers.Dense(16, activation='relu')(output)
    output = keras.layers.Dense(1, activation="sigmoid")(output)

    model = keras.models.Model(inputs=input, outputs=[output])

    model.compile(
        loss=keras.losses.binary_crossentropy,
        optimizer="rmsprop",
        metrics=["accuracy"]
    )
    model.summary()

    history = model.fit(x, y, batch_size=128, epochs=10, validation_split=0.2)

def process_cate_type_dnn5():
    df = pd.read_csv("data/covtype_onehot.csv")
    df = shuffle(df).reset_index(drop=True)

    from sklearn.preprocessing import StandardScaler

    array = df.values.astype(np.float32)
    scaler = StandardScaler()
    array[:, :10] = scaler.fit_transform(array[:, :10])

    print(array[:10,:15])

    x = array[:,:-1]
    y = array[:, -1]

    print(x.shape)
    print(y.shape)

    embed_reg = 0

    input = keras.layers.Input(shape=x.shape[1:])
    output = keras.layers.Dense(64, activation='relu')(input)
    output = keras.layers.Dense(32, activation='relu')(output)
    output = keras.layers.Dense(16, activation='relu')(output)
    output = keras.layers.Dense(1, activation="sigmoid")(output)

    model = keras.models.Model(inputs=input, outputs=[output])

    model.compile(
        loss=keras.losses.binary_crossentropy,
        optimizer="rmsprop",
        metrics=["accuracy"]
    )
    model.summary()

    history = model.fit(x, y, batch_size=128, epochs=10, validation_split=0.2)


if __name__ == '__main__':

    # not_process_cate_type()
    process_cate_type()
    # process_cate_type_dnn2()
