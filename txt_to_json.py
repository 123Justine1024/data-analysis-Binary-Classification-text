import json

# 打开文件并读取内容
with open("spamSNS.txt", "r", encoding="utf-8") as file:
    lines = file.readlines()

# 存储提取结果
data_list = []

# 解析每行数据
for line in lines:
    # 按制表符或多个空格分割
    parts = line.strip().split(maxsplit=2)  # 最多分为三部分
    if len(parts) == 3:
        item_id = int(parts[0])  # 编号
        label = int(parts[1])    # 标签
        content = parts[2]       # 内容
        # 添加到结果列表
        data_list.append({"number": item_id, "label": label, "text": content})

# 打印提取结果
for item in data_list:
    print(item)


with open("output.json", "w", encoding="utf-8") as outfile:
    json.dump(data_list, outfile, ensure_ascii=False, indent=4)
