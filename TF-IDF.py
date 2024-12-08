from sklearn.feature_extraction.text import TfidfVectorizer

# 示例文本
documents = ["有原因不明的泌尿系统结石等", "带给我们大常州一场壮观的视觉盛宴"]

# 初始化TF-IDF向量器
vectorizer = TfidfVectorizer()

# 生成TF-IDF矩阵
tfidf_matrix = vectorizer.fit_transform(documents)
print(tfidf_matrix)
# 转为密集矩阵（可选）
tfidf_dense = tfidf_matrix.toarray()
