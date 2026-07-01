import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 设置设备为 CPU
device = torch.device("cpu")
print(f"Using device: {device}")

# 自定义数据集类
class HandLandmarkDataset(Dataset):
    def __init__(self, features, labels):
        # 转换特征和标签为张量
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        # 返回数据集长度
        return len(self.labels)

    def __getitem__(self, idx):
        # 根据索引获取特征和标签
        return self.features[idx], self.labels[idx]

# 定义神经网络模型
class HandLandmarkClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(HandLandmarkClassifier, self).__init__()
        # 定义全连接层、批量归一化层和 Dropout 层
        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        # 前向传播过程
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# 读取数据集
csv_file_path = r'C:\Users\haha2\Desktop\CS\ASL Hand\runs\dataset_info.csv'
data = pd.read_csv(csv_file_path)

# 提取特征和标签
features = data.iloc[:, 2:].values
labels = data['class_index'].values

# 对标签进行编码
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(features, encoded_labels, test_size=0.2, random_state=42)

# 创建数据集对象
train_dataset = HandLandmarkDataset(X_train, y_train)
test_dataset = HandLandmarkDataset(X_test, y_test)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 获取输入和输出维度
input_size = features.shape[1]
num_classes = len(label_encoder.classes_)

# 初始化模型并移至指定设备
model = HandLandmarkClassifier(input_size, num_classes).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 定义学习率调度器
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10)

# 初始化最佳准确率和早停计数器
best_val_acc = 0
early_stopping_patience = 50
early_stopping_counter = 0

# 定义梯度累积步数
accumulation_steps = 4

# 记录训练开始时间
total_start_time = time.time()

# 开始训练
num_epochs = 5000
for epoch in range(num_epochs):
    start_time = time.time()
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()
    for i, (inputs, labels) in enumerate(train_loader):
        # 前向传播
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss = loss / accumulation_steps
        # 反向传播
        loss.backward()
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        running_loss += loss.item()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            # 验证集评估
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # 计算验证集准确率
    val_acc = 100 * correct / total

    end_time = time.time()
    epoch_time = end_time - start_time

    # 调整学习率
    prev_lr = optimizer.param_groups[0]['lr']
    scheduler.step(val_acc)
    new_lr = optimizer.param_groups[0]['lr']
    if new_lr != prev_lr:
        print(f'Learning rate adjusted from {prev_lr} to {new_lr} at epoch {epoch + 1}')

    # 打印训练信息
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.6f}, Val Acc: {val_acc:.3f}%, Time: {epoch_time:.2f}s')

    # 更新最佳准确率和早停计数器
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        early_stopping_counter = 0
        torch.save(model, 'best_model.pth')
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= early_stopping_patience:
            print(f'Early stopping at epoch {epoch + 1}')
            break

# 记录训练结束时间
total_end_time = time.time()
total_time = total_end_time - total_start_time

# 保存最后一个 epoch 的模型
torch.save(model, 'last_model.pth')
# 打印最佳准确率和总训练时间
print(f'Best Accuracy on test set: {best_val_acc}%')
print(f'Total training time: {total_time:.2f}s')