import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, recall_score, f1_score

# 读取数据
data = pd.read_csv(r'D:\document\MCM\week2\第二周小测数据\deduplicated_data.csv')

# 数据预处理
data['Detection Date'] = pd.to_datetime(data['Detection Date'])
data['Month'] = data['Detection Date'].dt.month  # 提取月份

# 标签编码
label_encoder = LabelEncoder()
data['Lab Status'] = label_encoder.fit_transform(data['Lab Status'])

# 特征和目标变量
X = data[['Month']]
y = data['Lab Status']

# 划分训练集和测试集（8:2比例）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 构建逻辑回归模型
model = LogisticRegression()
model.fit(X_train, y_train)

# 使用测试集进行预测
y_pred = model.predict(X_test)

# 计算评估指标
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred, average='weighted')  # 使用加权平均
f1 = f1_score(y_test, y_pred, average='weighted')  # 使用加权平均

# 输出评估结果
print(f'Accuracy: {accuracy*100:.6f}%')
print(f'Recall: {recall*100:.4f}%')
print(f'F1 Score: {f1:.4f}')