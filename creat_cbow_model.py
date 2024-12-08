import json
import jieba
from gensim.models import Word2Vec


# 加载数据
with open('output.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

texts = [item["text"] for item in data]

# 使用 jieba 分词
tokenized_texts = [list(jieba.cut(text)) for text in texts]
print("分词结果:", tokenized_texts)

# 训练 CBOW 模型
cbow_model = Word2Vec(
    sentences=tokenized_texts,  # 分词后的语料
    vector_size=100,           # 词向量维度
    window=5,                  # 上下文窗口大小
    min_count=1,               # 词频阈值，低于该值的词将被忽略
    sg=0,                      # 使用 CBOW 模型（sg=0 表示 CBOW，sg=1 表示 Skip-Gram）
    workers=4                  # 多线程加速训练
)
# 保存模型
cbow_model.save("cbow_model.model")

# 加载模型
loaded_model = Word2Vec.load("cbow_model.model")

# 查看模型中的词汇表
print("词汇表:", cbow_model.wv.index_to_key)

# 获取特定词的词向量
vector = cbow_model.wv["深度学习"]
print("词向量（深度学习）:", vector)

