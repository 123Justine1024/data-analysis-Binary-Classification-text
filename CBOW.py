import json

import jieba
from gensim.models import Word2Vec

import numpy as np


# 将句子转换为词向量矩阵
def sentence_to_vector(sentence, word2vec_model, max_len, vector_size):
    vectors = []
    for word in sentence:
        if word in word2vec_model.wv:
            vectors.append(word2vec_model.wv[word])
        else:
            vectors.append(np.zeros(vector_size))  # 未登录词用零向量表示
    # 截断或填充
    if len(vectors) > max_len:
        vectors = vectors[:max_len]
    else:
        vectors += [np.zeros(vector_size)] * (max_len - len(vectors))
    return np.array(vectors)


# 加载模型
cbow_model = Word2Vec.load("cbow_model.model")

# 示例
max_len = 23  # 句子最大长度
vector_size = 100  # CBOW 生成的词向量维度
# 加载数据
with open('output.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

texts = [item["text"] for item in data]


# 使用 jieba 分词
tokenized_texts = [list(jieba.cut(text)) for text in texts]
print(tokenized_texts)

x_data = np.array([sentence_to_vector(sentence, cbow_model, max_len, vector_size) for sentence in tokenized_texts])

import h5py

# 保存为 HDF5 文件
with h5py.File("x_data.h5", "w") as hf:
    hf.create_dataset("x_data", data=x_data)
print("x_data 已保存为 x_data.h5")


print("句子向量矩阵形状:", x_data)  # (num_sentences, max_len, vector_size)
