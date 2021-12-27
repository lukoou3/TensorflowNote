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

    cate_cols = dense_features + sparse_features
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


    x = df.iloc[:,:-1].values.astype(np.int32)
    y = df.iloc[:,-1].values.astype(np.float64)

    return x, y, cate_size

class FMLayer(keras.layers.Layer):
    """
    https://www.jianshu.com/p/152ae633fb00
    https://zhuanlan.zhihu.com/p/58160982
    https://blog.csdn.net/hiwallace/article/details/81333604
    https://www.zhihu.com/question/352399723, 数值类型的特征怎么加入深度模型如nfm，deepfm?
    """

    def __init__(self, sparse_cate_size, k = 8, w_reg=1e-6, v_reg=1e-6, **kwargs):
        # input_shape一般通过kwargs传入
        super().__init__(**kwargs)
        self.sparse_cate_size = sparse_cate_size
        self.k = k
        self.w_reg = w_reg
        self.v_reg = v_reg


    def build(self, input_shape):
        """构建所需要的参数"""
        print(input_shape)
        self.w0 = self.add_weight(name='w0',
                                       shape=(1, ),
                                       initializer='zeros',
                                       trainable=True)
        self.w = self.add_weight(name='w',
                                        shape=(self.sparse_cate_size, 1),
                                        initializer='uniform',
                                        regularizer=l2(self.w_reg),
                                        trainable=True)
        self.V = self.add_weight(name='V', shape=(self.sparse_cate_size, self.k),
                                 initializer='uniform',
                                 regularizer=l2(self.v_reg),
                                 trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        """完成正向计算"""
        sparse_input = inputs

        # 线性部分
        weight = tf.nn.embedding_lookup(self.w, sparse_input)  # (batch_size, fields, 1)
        y_first_order = self.w0 + tf.reduce_sum(weight, axis=1)  # (batch_size, 1)

        # 交叉部分
        second_inputs = tf.nn.embedding_lookup(self.V, sparse_input)  # (batch_size, fields, k)
        square_sum = tf.reduce_sum(second_inputs, axis=1) ** 2 # (batch_size, k)
        sum_square = tf.reduce_sum(second_inputs ** 2, axis=1) # (batch_size, k)
        y_second_order = 0.5 * tf.reduce_sum(square_sum - sum_square, axis=1, keepdims=True)  # (batch_size, 1)

        return y_first_order + y_second_order

def test():
    x, y, cate_size = get_data()

    # 这里还必须声明dtype为int32, layers.Embedding不用转是因为它的call函数中会判断类型不是int会转换: inputs = math_ops.cast(inputs, 'int32')
    input = keras.layers.Input(shape=[x.shape[-1]], dtype='int32')
    fm_output = FMLayer(cate_size, k = 8)(input)
    output = keras.layers.Lambda(lambda x: tf.nn.sigmoid(x))(fm_output)

    model = keras.models.Model(inputs=[input], outputs=[output])

    model.compile(loss="binary_crossentropy",
                  optimizer = "adam",
                  metrics = ["accuracy"])

    model.summary()

    history = model.fit(x, y, batch_size=4096, epochs=20, validation_split=0.2)


if __name__ == '__main__':
    test()
