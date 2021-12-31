import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from sklearn.utils import shuffle
import collections
import random

def get_data():
    df_user = pd.read_csv("../data/ml-1m/users.dat",
                          sep="::", header=None, engine="python",
                          names=["user_id", "gender", "age", "occupation", "zip_code"])
    df_movie = pd.read_csv("../data/ml-1m/movies.dat",
                           sep="::", header=None, engine="python",
                           names=["movie_id", "title", "genres"])
    df_rating = pd.read_csv("../data/ml-1m/ratings.dat",
                            sep="::", header=None, engine="python",
                            names=["user_id", "movie_id", "rating", "ts"])#.head(100000)

    neg_cnt = 1
    all_movie_id = set(df_rating["movie_id"].unique())
    def get_net_movies(s):
        array = all_movie_id - set(s)
        size = neg_cnt * len(s)
        return random.sample(array, min(size, len(array)))

    neg_df_rating = df_rating.groupby("user_id")["movie_id"].apply(get_net_movies).reset_index().explode('movie_id').assign(label = 0)
    df_label = pd.concat([df_rating[["user_id", "movie_id"]].assign(label = 1), neg_df_rating])

    # 计算电影中每个题材的次数
    genre_count = collections.defaultdict(int)
    for genres in df_movie["genres"].str.split("|"):
        for genre in genres:
            genre_count[genre] += 1
    print(genre_count)

    # 只保留最有代表性的题材
    def get_highrate_genre(x):
        sub_values = {}
        for genre in x.split("|"):
            sub_values[genre] = genre_count[genre]
        return sorted(sub_values.items(), key=lambda x:x[1], reverse=True)[0][0]
    df_movie["genres"] = df_movie["genres"].map(get_highrate_genre)

    # 合并成一个df
    df = pd.merge(pd.merge(df_label, df_user), df_movie)
    df.drop(columns=["ts", "zip_code", "title"], inplace=True, errors='ignore')

    cate_cols = ["user_id", "gender", "age", "occupation", "movie_id", "genres"]
    cate_map = {}
    size = 0
    for col in cate_cols:
        cate_map[col] = {v: i for i, v in enumerate(df[col].unique())}
        size += len(cate_map[col])
    for col in cate_cols:
        df[col] = df[col].map(cate_map[col])

    cate_size = sum([len(map) for col, map in cate_map.items()])
    # print(cate_map)
    print(cate_size)


    df = shuffle(df).reset_index(drop=True)
    return df

def test():
    """
    https://zhuanlan.zhihu.com/p/136253355: 实践DSSM召回模型
    https://www.jianshu.com/p/7d4c65a66cac: 推荐系统论文阅读（七)-借鉴DSSM构建双塔召回模型
    https://github.com/shenweichen/DeepMatch: 大佬用tf实现的一些召回模型
    :return:
    """
    df = get_data()

    print(df.label.value_counts())

    # 输入
    input_user_id = keras.layers.Input(shape=(1,), dtype='int32', name="user_id")
    input_gender = keras.layers.Input(shape=(1,), dtype='int32', name="gender")
    input_age = keras.layers.Input(shape=(1,), dtype='int32', name="age")
    input_occupation = keras.layers.Input(shape=(1,), dtype='int32', name="occupation")
    input_movie_id = keras.layers.Input(shape=(1,), dtype='int32', name="movie_id")
    input_genre = keras.layers.Input(shape=(1,), dtype='int32', name="genre")

    embedding_user_id = keras.layers.Embedding(df['user_id'].max() + 1, 8, embeddings_regularizer=l2(1e-6))(input_user_id) # [None, 1, embed_dim]
    embedding_gender = keras.layers.Embedding(df['gender'].max() + 1, 2, embeddings_regularizer=l2(1e-6))(input_gender) # [None, 1, embed_dim]
    embedding_ager = keras.layers.Embedding(df['age'].max() + 1, 2, embeddings_regularizer=l2(1e-6))(input_age) # [None, 1, embed_dim]
    embedding_occupation = keras.layers.Embedding(df['occupation'].max() + 1, 2, embeddings_regularizer=l2(1e-6))(input_occupation) # [None, 1, embed_dim]
    user_embedding = keras.layers.concatenate([embedding_user_id, embedding_gender, embedding_ager, embedding_occupation])# [None, 1, user_embed_dim]
    user_embedding = keras.layers.Reshape([-1])(user_embedding) # [None, user_embed_dim]
    user_embedding = keras.layers.Dense(32, activation='relu', kernel_regularizer=l2(1e-6))(user_embedding)
    user_embedding = keras.layers.Dense(16, activation='relu', name="user_embedding", kernel_regularizer=l2(1e-6))(user_embedding)

    embedding_movie_id = keras.layers.Embedding(df['movie_id'].max() + 1, 8, embeddings_regularizer=l2(1e-6))(input_movie_id) # [None, 1, embed_dim]
    embedding_genre = keras.layers.Embedding(df['genres'].max() + 1, 2, embeddings_regularizer=l2(1e-6))(input_genre) # [None, 1, embed_dim]
    movie_embedding = keras.layers.concatenate([embedding_movie_id, embedding_genre])# [None, 1, user_embed_dim]
    movie_embedding = keras.layers.Reshape([-1])(movie_embedding) # [None, movie_embed_dim]
    movie_embedding = keras.layers.Dense(32, activation='relu', kernel_regularizer=l2(1e-6))(movie_embedding)
    movie_embedding = keras.layers.Dense(16, activation='relu', name="movie_embedding", kernel_regularizer=l2(1e-6))(movie_embedding)

    def cosine_layer(inputs):
        a = inputs[0]
        b = inputs[1]
        a_norm = tf.norm(a, axis=1, keepdims=True)
        b_norm = tf.norm(b, axis=1, keepdims=True)
        return tf.clip_by_value(tf.reduce_sum(a * b, axis=1, keepdims=True) / (a_norm * b_norm + 1e-8), -1, 1.0)

    output = keras.layers.Lambda(cosine_layer)([user_embedding, movie_embedding])
    # output = keras.layers.Lambda(lambda x: tf.sigmoid(x))(output)
    output = keras.layers.Dense(1, activation="sigmoid")(output)

    model = keras.models.Model(inputs=[input_user_id, input_gender, input_age, input_occupation, input_movie_id, input_genre], outputs=[output])

    model.compile(
        loss=keras.losses.binary_crossentropy,
        # optimizer=keras.optimizers.SGD(),
        optimizer=keras.optimizers.Adam(),
        metrics=["accuracy", keras.metrics.AUC(name = 'auc')]
    )
    model.summary()

    history = model.fit([df['user_id'], df['gender'], df['age'], df['occupation'], df['movie_id'], df['genres']], df['label'],
                        batch_size=256, epochs=10, validation_split=0.1)

if __name__ == '__main__':
    test()