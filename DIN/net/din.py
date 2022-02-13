import os
import numpy as np
import pandas as pd
from collections import namedtuple

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# 使用具名元组定义特征标记
SparseFeature = namedtuple('SparseFeature', ['name', 'vocabulary_size', 'embedding_size'])
DenseFeature = namedtuple('DenseFeature', ['name', 'dimension'])
VarLenSparseFeature = namedtuple('VarLenSparseFeature', ['name', 'vocabulary_size', 'embedding_size', 'maxlen'])

##### 数据预处理
data = pd.read_csv('./data/movie_sample.txt', sep="\t", header=None)
data.columns = ["user_id", "gender", "age", "hist_movie_id", "hist_len", "movie_id", "movie_type_id", "label"]
data.head()

# 输入数据
X = data[['user_id', 'gender', 'age', 'hist_movie_id', 'hist_len',
          'movie_id', 'movie_type_id']]
y = data['label']
X_train = {
    'user_id': np.array(X['user_id']),
    'gender': np.array(X['gender']),
    'age': np.array(X['age']),
    'hist_movie_id': np.array([[int(i) for i in s.split(',')] for s in X['hist_movie_id']]),
    'hist_len': np.array(X['hist_len']),
    'movie_id': np.array(X['movie_id']),
    'movie_type_id': np.array(X['movie_type_id'])
}
y_train = np.array(y)


def build_input_layers(feature_columns):
    """ 构建输入层 """
    input_layer_dict = {}
    for f in feature_columns:
        if isinstance(f, DenseFeature):
            input_layer_dict[f.name] = Input(shape=(f.dimension,), name=f.name)
        elif isinstance(f, SparseFeature):
            input_layer_dict[f.name] = Input(shape=(1,), name=f.name)
        elif isinstance(f, VarLenSparseFeature):
            input_layer_dict[f.name] = Input(shape=(f.maxlen,), name=f.name)
    return input_layer_dict


def build_embedding_layers(feature_columns):
    embedding_layers_dict = {}
    for f in feature_columns:
        if isinstance(f, SparseFeature):
            embedding_layers_dict[f.name] = Embedding(f.vocabulary_size + 1, f.embedding_size, name='emb_' + f.name)
        elif isinstance(f, VarLenSparseFeature):
            embedding_layers_dict[f.name] = Embedding(f.vocabulary_size + 1, f.embedding_size, name='var_emb_' + f.name,
                                                      mask_zero=True)
    return embedding_layers_dict


def concat_embedding_list(feature_columns, input_layer_dict, embedding_layer_dict, flatten=False):
    """ 拼接embedding特征 """
    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeature), feature_columns)) if feature_columns else []
    embedding_list = []
    for f in sparse_feature_columns:
        _input_layer = input_layer_dict[f.name]
        _embed = embedding_layer_dict[f.name]
        embed_layer = _embed(_input_layer)
        if flatten:
            embed_layer = Flatten()(embed_layer)

        embedding_list.append(embed_layer)
    return embedding_list


def embedding_lookup(feature_columns, input_layer_dict, embedding_layer_dict):
    embedding_list = []
    for f in feature_columns:
        _input_layer = input_layer_dict[f]
        _embed = embedding_layer_dict[f]
        embed_layer = _embed(_input_layer)
        embedding_list.append(embed_layer)
    return embedding_list


def concat_input_list(input_list):
    """ 合并input列表 """
    _nums = len(input_list)
    if _nums > 1:
        return Concatenate(axis=1)(input_list)
    elif len(input_list) == 1:
        return input_list[0]
    else:
        return None

class Dice(Layer):
    def __init__(self):
        super(Dice, self).__init__()
        self.bn = BatchNormalization(center=False, scale=False)
        self.alpha = self.add_weight(shape=(), dtype=tf.float32, name='alpha')

    def call(self, x):
        x_normed = self.bn(x)
        x_p = tf.sigmoid(x_normed)

        return self.alpha * (1.0 - x_p) * x + x_p * x

class LocalActivationUnit(Layer):

    def __init__(self, hidden_units=(256, 128, 56), activation='prelu'):
        super(LocalActivationUnit, self).__init__()
        self.hidden_units = hidden_units
        self.linear = Dense(1)
        self.dnn = [Dense(unit, activation=PReLU() if activation == 'prelu' else Dice()) for unit in hidden_units]

    def call(self, inputs):
        query, keys = inputs

        # 序列长度
        keys_len = keys.get_shape()[1]
        queries = tf.tile(query, multiples=[1, keys_len, 1])  # B x keys_len x emb_size

        # 对特征进行拼接（原始向量、向量差、外积结果）
        attention_input = tf.concat([queries, keys, queries - keys, queries * keys],
                                    axis=-1)  # B x keys_len x 4*emb_size

        # 将原始向量和
        attention_out = attention_input
        for fc in self.dnn:
            attention_out = fc(attention_out)  # B x keys_len x 56

        attention_out = self.linear(attention_out)  # B x keys_len x 1
        attention_out = tf.squeeze(attention_out, -1)  # B x keys_len
        return attention_out


class AttentionPoolingLayer(Layer):
    def __init__(self, attention_hidden_units=(256, 128, 56)):
        super(AttentionPoolingLayer, self).__init__()
        self.attention_hidden_units = attention_hidden_units
        # 对输入做了一层转化
        self.local_attention = LocalActivationUnit(self.attention_hidden_units)

    def call(self, inputs):
        # queries: B x 1 x emb_size keys: B x keys_len x emb_size
        queries, keys = inputs

        # pad在计算atten_score也有得分，这四步的作用是借助padding的embedding为0，将atten_score中padding部分的兴趣分转化为0。

        # 获取行为序列中每个商品对应的注意力权重
        attention_score = self.local_attention([queries, keys])

        # 获取行为序列embedding的mask矩阵，将embedding矩阵非零元素设置为True
        key_masks = tf.not_equal(keys[:, :, 0], 0)  # B x keys_len

        # 标记行为序列embedding中无效的位置
        paddings = tf.zeros_like(attention_score)  # B x keys_len

        outputs = tf.where(key_masks, attention_score, paddings)  # B x 1 x keys_len
        outputs = tf.expand_dims(outputs, axis=1)  # B x 1 x keys_len
        outputs = tf.matmul(outputs, keys)  # B x 1 x emb_size
        outputs = tf.squeeze(outputs, axis=1)  # B x emb_size
        return outputs


def get_dnn_logits(dnn_input, hidden_units=(200, 80), activation='prelu'):
    dnn_list = [Dense(unit, activation=PReLU() if activation == 'prelu' else Dice()) for unit in hidden_units]
    dnn_out = dnn_input
    for dnn in dnn_list:
        dnn_out = dnn(dnn_out)

    dnn_logits = Dense(1, activation='sigmoid')(dnn_out)
    return dnn_logits


def DIN(feature_columns, behavior_feature_list, behavior_seq_feature_list):
    """ Deep Interest Network，采用注意力机制对用户的兴趣动态模拟 """
    input_layer_dict = build_input_layers(feature_columns)
    input_layers = list(input_layer_dict.values())

    # 特征中的sparse和dense特征
    sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeature), feature_columns))
    dense_feature_columns = list(filter(lambda x: isinstance(x, DenseFeature), feature_columns))

    # dense
    dnn_dense_input_list = []
    for f in dense_feature_columns:
        dnn_dense_input_list.append(input_layer_dict[f.name])

    # dense特征拼接
    concat_dnn_dense_input = concat_input_list(dnn_dense_input_list)

    # embedding
    embedding_layer_dict = build_embedding_layers(feature_columns)

    dnn_sparse_embed_input = concat_embedding_list(sparse_feature_columns, input_layer_dict, embedding_layer_dict,
                                                   flatten=True)
    print('dnn_sparse_embed_input: ', dnn_sparse_embed_input)
    concat_dnn_sparse_embed_input = concat_input_list(dnn_sparse_embed_input)

    # print('input_layer_dict: {}, embedding_layer_dict: {}'.format(input_layer_dict, embedding_layer_dict))

    # 获取当前物品movie的embedding
    query_embed_list = embedding_lookup(behavior_feature_list, input_layer_dict, embedding_layer_dict)
    # 获取行为序列的embedding
    key_embed_list = embedding_lookup(behavior_seq_feature_list, input_layer_dict, embedding_layer_dict)

    # 使用注意力机制将历史movie_id进行池化
    dnn_seq_input_list = []
    for i in range(len(key_embed_list)):
        # # B x emb_size
        seq_emb = AttentionPoolingLayer()([query_embed_list[i], key_embed_list[i]])
        dnn_seq_input_list.append(seq_emb)

    # 将多个行为序列的attention pooling（当前物品和行为序列物品的互操作求注意力权重，兴趣分 = 权重*历史行为物品的embedding）的embedding进行拼接
    concat_dnn_seq_input_list = concat_input_list(dnn_seq_input_list)

    # 将dense特征、sparse特征及注意力加权的序列特征拼接
    print('concat_dnn_dense_input: ', concat_dnn_dense_input)
    print('concat_dnn_sparse_embed_input: ', concat_dnn_sparse_embed_input)
    print('concat_dnn_seq_input_list: ', concat_dnn_seq_input_list)

    dnn_input = Concatenate(axis=1)([concat_dnn_dense_input, concat_dnn_sparse_embed_input, concat_dnn_seq_input_list])
    dnn_logits = get_dnn_logits(dnn_input, activation='prelu')
    model = Model(input_layers, dnn_logits)
    return model


# 特征列
feature_columns = [
    SparseFeature('user_id', data.user_id.max() + 1, embedding_size=8),
    SparseFeature('gender', data.gender.max() + 1, embedding_size=8),
    SparseFeature('age', data.age.max() + 1, embedding_size=8),
    SparseFeature('movie_id', data.movie_id.max() + 1, embedding_size=8),
    SparseFeature('movie_type_id', data.movie_type_id.max() + 1, embedding_size=8),
    DenseFeature('hist_len', 1),
]

feature_columns += [VarLenSparseFeature('hist_movie_id',
                                        vocabulary_size=data.movie_id.max() + 1,
                                        embedding_size=8,
                                        maxlen=50)]

# print('feature_columns: ', feature_columns)

# 行为特征列表 行为序列特征列表
behavior_feature_list = ['movie_id']
behavior_seq_feature_list = ['hist_movie_id']
model = DIN(feature_columns, behavior_feature_list, behavior_seq_feature_list)
model.summary()

model.compile(optimizer="adam",
              loss="binary_crossentropy",
              metrics=["binary_crossentropy", tf.keras.metrics.AUC(name='auc')])

model.fit(X_train, y_train,
          batch_size=64, epochs=5, validation_split=0.2)