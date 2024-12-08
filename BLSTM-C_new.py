import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Bidirectional, LSTM, Conv1D, MaxPooling1D, Flatten, Dense
from sklearn.model_selection import train_test_split

# 模型参数
sequence_length = 23  # 输入序列长度
input_dim = 100         # 输入特征维度
lstm_units = 64        # LSTM单元数
cnn_filters = 64       # 卷积层的过滤器数量
cnn_kernel_size = 7    # 卷积核大小
pool_size = 5          # 池化窗口大小
num_classes = 2       # 输出分类数量

# 输入层
inputs = Input(shape=(sequence_length, input_dim), name="Input_Layer")

# BLSTM层
blstm = Bidirectional(LSTM(units=lstm_units, return_sequences=True), name="BLSTM_Layer")(inputs)

# CNN层
cnn = Conv1D(filters=cnn_filters, kernel_size=cnn_kernel_size, activation='relu', name="CNN_Layer")(blstm)

# 最大池化层
pool = MaxPooling1D(pool_size=pool_size, name="MaxPooling_Layer")(cnn)

# 展平层
flatten = Flatten(name="Flatten_Layer")(pool)

# 输出层
outputs = Dense(num_classes, activation='softmax', name="Output_Layer")(flatten)

# 构建模型
model = Model(inputs=inputs, outputs=outputs, name="BLSTM_C_Model")

# 模型摘要
model.summary()

# 编译模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 加载数据
with h5py.File("x_data.h5", "r") as hf:
    x_data_loaded = hf["x_data"][:]
# 读取数据
y_One_Hot_Encoding = np.load("y_One_Hot_Encoding.npy")
print(len(x_data_loaded))
print(len(y_One_Hot_Encoding))

x_train, x_test, y_train, y_test = train_test_split(x_data_loaded, y_One_Hot_Encoding, test_size=0.3, random_state=42)
print(2)

model.fit(x_train, y_train, batch_size=32, epochs=1, validation_split=0.2)
print(3)

import os

# 文件路径
file_path = "BLSTM-C_model_new_save.h5"

# 如果文件存在，则删除
if os.path.exists(file_path):
    os.remove(file_path)

# 保存模型
try:
    model.save(file_path)
    print(f"模型已成功保存到 {file_path}")
except Exception as e:
    print(f"保存模型时出错: {e}")

# 测试模型
loss, accuracy = model.evaluate(x_test, y_test, batch_size=32)

print(f"测试集损失: {loss}")
print(f"测试集准确率: {accuracy}")
