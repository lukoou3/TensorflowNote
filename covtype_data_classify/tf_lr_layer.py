import os
os.environ ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.regularizers import l2
from sklearn.utils import shuffle


class LrLayer(keras.layers.Layer):
    """支持输入稀疏类别特征的lr, 省去OneHot步骤, 节省内存"""

    def __init__(self, dense_col_size, sparse_col_size, sparse_cate_size, w_reg=1e-6, v_reg=1e-6, **kwargs):
        # input_shape一般通过kwargs传入
        super().__init__(**kwargs)
        self.dense_col_size = dense_col_size
        self.sparse_col_size = sparse_col_size
        self.sparse_cate_size = sparse_cate_size
        self.w_reg = w_reg
        self.v_reg = v_reg

    def build(self, input_shape):
        """构建所需要的参数"""
        print(input_shape)
        self.w_dense = self.add_weight(name='w_dense',
                                       shape=(self.dense_col_size, 1),
                                       initializer='uniform',
                                       regularizer=l2(self.w_reg),
                                       trainable=True)
        self.w_sparse = self.add_weight(name='w_sparse',
                                        shape=(self.sparse_cate_size, 1),
                                        initializer='uniform',
                                        regularizer=l2(self.v_reg),
                                        trainable=True)
        # 偏置/截距一般初始化为0
        self.bias = self.add_weight(name='bias',
                                    shape=(1,),
                                    initializer='zeros',
                                    trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        """完成正向计算"""
        dense_input, sparse_input = inputs
        sparse_inputs = tf.nn.embedding_lookup(self.w_sparse, sparse_input)  # (batch_size, fields, embed_dim)
        w = tf.reduce_sum(sparse_inputs, axis=1, keepdims=False)  # (batch_size, 1, embed_dim)
        return tf.nn.sigmoid(dense_input @ self.w_dense + w + self.bias)


def test():
    df = pd.read_csv("data/covtype_scaler.csv")
    # 这里必须shuffle, 文件中的数据似乎很多都是同类别的连在一起, 不shuffle训练效果很差
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
    x_spase = array[:, -3:-1].astype(np.int32)
    y = array[:, -1]

    input_dense = keras.layers.Input(shape=x_dense.shape[1:])
    input_spase = keras.layers.Input(shape=x_spase.shape[1:], dtype="int32")
    output = LrLayer(x_dense.shape[-1], x_spase.shape[-1], cate_size)([input_dense, input_spase])

    model = keras.models.Model(inputs=[input_dense, input_spase], outputs=[output])

    model.compile(
        loss=keras.losses.binary_crossentropy,
        optimizer=keras.optimizers.SGD(),
        metrics=["accuracy"]
    )
    model.summary()

    history = model.fit([x_dense, x_spase], y, batch_size=128, epochs=10, validation_split=0.2)


if __name__ == '__main__':
    test()
