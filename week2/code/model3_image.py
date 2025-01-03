import torch
import torchvision
from torchvision import transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

# 配置设备
device = torch.device("cuda")

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载数据集
train_dataset = torchvision.datasets.ImageFolder(root=r'D:\document\MCM\week2\第二周小测图片文件\train_data', transform=transform)
val_dataset = torchvision.datasets.ImageFolder(root=r'D:\document\MCM\week2\第二周小测图片文件\val_data', transform=transform)
test_dataset = torchvision.datasets.ImageFolder(root=r'D:\document\MCM\week2\第二周小测图片文件\test_data', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 计算类别权重
labels = [label for _, label in train_dataset.samples]  # 获取训练集所有标签
class_weights = compute_class_weight(class_weight='balanced', classes=[0, 1], y=labels)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
print(f"Class Weights: {class_weights}")  # 打印权重，检查是否合理

# 定义加权损失函数
criterion = nn.BCELoss(reduction='none')  # 不做自动加权
def weighted_loss(outputs, targets):
    weights = class_weights[targets.long()]
    loss = criterion(outputs, targets)
    return (loss * weights).mean()

# 加载预训练 ResNet-50 模型
model = models.resnet50(pretrained=True)

# 替换分类头
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 512),  # 添加一个隐藏层
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 1),  # 二分类输出
    nn.Sigmoid()  # 用 Sigmoid 输出概率
)

model = model.to(device)

# 定义优化器和学习率调整器
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

# 早停参数
early_stopping_patience = 10
best_val_f1 = 0.0
early_stopping_counter = 0

# 训练模型
epochs = 50
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    
    with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}") as pbar:
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.float().to(device)
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = weighted_loss(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.update(1)

    avg_train_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_train_loss:.6f}")

    # 验证模型
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        with tqdm(total=len(val_loader), desc="Validation") as pbar:
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.float().to(device)
                outputs = model(inputs).squeeze()
                loss = weighted_loss(outputs, labels)
                val_loss += loss.item()
                preds = (outputs > 0.5).int()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                pbar.update(1)

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = accuracy_score(all_labels, all_preds)
    val_recall = recall_score(all_labels, all_preds)
    val_f1 = f1_score(all_labels, all_preds)
    print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {avg_val_loss:.6f}, Validation Accuracy: {val_accuracy * 100:.2f}%, Validation Recall: {val_recall * 100:.2f}%, Validation F1: {val_f1:.4f}")

    # 调整学习率
    scheduler.step(avg_val_loss)

    # 早停机制
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), "best_model.pth")  # 保存最佳模型
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= early_stopping_patience:
            print("Early stopping triggered.")
            break

# 测试集评估
model.load_state_dict(torch.load("best_model.pth"))  # 加载验证集上表现最佳的模型
model.eval()

test_loss = 0.0
all_preds = []
all_labels = []
with torch.no_grad():
    with tqdm(total=len(test_loader), desc="Testing") as pbar:
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.float().to(device)
            outputs = model(inputs).squeeze()
            loss = weighted_loss(outputs, labels)
            test_loss += loss.item()
            preds = (outputs > 0.5).int()  # 将概率转为0或1
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            pbar.update(1)

# 计算指标
avg_test_loss = test_loss / len(test_loader)
test_accuracy = accuracy_score(all_labels, all_preds)
test_recall = recall_score(all_labels, all_preds)
test_f1 = f1_score(all_labels, all_preds)

# 打印测试结果
print(f"Test Loss: {avg_test_loss:.6f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Recall: {test_recall * 100:.2f}%")
print(f"Test F1 Score: {test_f1:.4f}")
