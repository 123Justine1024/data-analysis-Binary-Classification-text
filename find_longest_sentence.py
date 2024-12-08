import json
import jieba
from gensim.models import Word2Vec

# 加载数据
with open('output.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
texts = [item["text"] for item in data]
max_long_text = 0
load = -1
max_text = ""
i = 0
for text in texts:
    print(i)
    i = i + 1
    tokenized_texts = list(jieba.cut(text))
    if len(tokenized_texts) > max_long_text:
        max_long_text = len(tokenized_texts)
        max_text = text

print(max_long_text)
print(max_text)
