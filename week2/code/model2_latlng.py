import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score

# 1. 加载数据
data = pd.read_csv(r'D:\document\MCM\week2\第二周小测数据\deduplicated_data.csv') # 你的数据文件

# 2. 提取特征和目标变量
features = data[['Latitude', 'Longitude']] # 经纬度作为特征
target = data['Lab Status'] # 实验室状态作为目标

# 3. 标签编码 Lab Status（如果它是类别变量）
label_encoder = LabelEncoder()
target = label_encoder.fit_transform(target) # 转换为数字编码

# 4. 数据标准化
scaler = StandardScaler()
features = scaler.fit_transform(features) # 标准化经纬度

# 5. 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# 6. 设置固定的 n_neighbors 值，创建 KNN 分类器
n_neighbors = 5 # 你可以调整这个值
knn = KNeighborsClassifier(n_neighbors=n_neighbors)

# 7. 训练模型
knn.fit(X_train, y_train)

# 8. 在测试集上进行预测
y_pred = knn.predict(X_test)

# 9. 评估模型
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy with n_neighbors={n_neighbors}: {accuracy * 100:.2f}%")

# 10. 计算 Recall 和 F1 分数
recall = recall_score(y_test, y_pred, average='weighted') # weighted 适用于多类分类
f1 = f1_score(y_test, y_pred, average='weighted') # weighted 适用于多类分类

print(f"Recall with n_neighbors={n_neighbors}: {recall * 100:.2f}%")
print(f"F1-Score with n_neighbors={n_neighbors}: {f1 * 100:.2f}%")

# 11. 假设你想预测一个新的经纬度
new_latitude = 39.7392 # 输入的经纬度
new_longitude = -104.9903

# 12. 处理新的输入数据
new_data = pd.DataFrame({'Latitude': [new_latitude], 'Longitude': [new_longitude]})
new_data = scaler.transform(new_data) # 标准化新的输入数据

# 13. 进行预测
predicted_status = knn.predict(new_data) # 预测实验室状态

# 14. 输出预测结果
predicted_status_label = label_encoder.inverse_transform(predicted_status) # 将预测结果转换为原始标签
print(f"Predicted Lab Status for latitude {new_latitude}, longitude {new_longitude}: {predicted_status_label[0]}")
