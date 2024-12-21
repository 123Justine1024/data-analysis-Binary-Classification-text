from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import json
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load

# 加载数据
with open('output.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

texts = [item["text"] for item in data]
labels = [item["label"] for item in data]

# 使用TF-IDF向量化
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)
print(X)
y = labels

# 将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建加权随机森林分类器
clf = RandomForestClassifier(n_estimators=1, class_weight="balanced", random_state=42, verbose=1)

# 分批训练
batch_size = 100
num_batches = int(np.ceil(X_train.shape[0] / batch_size))

for i in tqdm(range(num_batches), desc="Training Batches"):
    start_idx = i * batch_size
    end_idx = min((i + 1) * batch_size, X_train.shape[0])
    clf.fit(X_train[start_idx:end_idx], y_train[start_idx:end_idx])

dump(clf, 'random_forest_model.joblib')
# 预测
y_pred = clf.predict(X_test)

# 输出准确率
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
