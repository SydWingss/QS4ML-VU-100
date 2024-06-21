# 导入必要的库
import os, datetime
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from rich.progress import Progress
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

# 加载数据
train_data = pd.read_csv('Dataset/train.csv')
test_data = pd.read_csv('Dataset/test.csv')

# 分割特征和标签
train_features = train_data.drop('act', axis=1)
test_features = test_data.drop('act', axis=1)
train_labels = train_data['act']
test_labels = test_data['act']

# 数据预处理
# scaler = StandardScaler()
# train_features = scaler.fit_transform(train_features)
# test_features = scaler.transform(test_features)

# 定义数据集
class Data(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)

train_data = Data(torch.FloatTensor(train_features.to_numpy()), torch.FloatTensor(train_labels.to_numpy()))
test_data = Data(torch.FloatTensor(test_features.to_numpy()), torch.FloatTensor(test_labels.to_numpy()))

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.relu = nn.ReLU()
        if in_channels != out_channels:
            self.linear_shortcut = nn.Linear(in_channels, out_channels)
        else:
            self.linear_shortcut = nn.Identity()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        out = self.linear(x)
        out = self.relu(out)
        out = self.dropout(out)
        shortcut = self.linear_shortcut(x)
        out += shortcut
        return out

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer = nn.Sequential(
            ResidualBlock(train_features.shape[1], 128),  # 增加每层的单元数
            ResidualBlock(128, 128),
            ResidualBlock(128, 64),
            ResidualBlock(64, 32),
            nn.Dropout(0.5),
            nn.Linear(32, 4)  # num_classes是你的类别数量
        )
        
    def forward(self, x):
        x = self.layer(x)
        return nn.functional.softmax(x, dim=1)

# 检查是否支持CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义超参数
for i in range(5):
    for j in range(8):
        for k in range(5):
            for l in range(6):
                EPOCHS = 200 * (i + 1)
                BATCH_SIZE = 128 * (2**(j + 1))
                LEARNING_RATE = 1e-7 * (10**k)
                WEIGHT_DECAY = 1e-6 * (10**l)

                # 创建模型并移动到GPU上
                model = Net().to(device)

                # 定义损失函数和优化器
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

                # 初始化进度条
                progress = Progress()

                # 训练模型
                losses = []
                accuracies = []

                with progress:
                    task1 = progress.add_task("[cyan]Training...", total=EPOCHS)
                    for epoch in range(EPOCHS):
                        running_loss = 0.0
                        correct = 0
                        total = 0
                        for X, y in DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True):
                            # 将数据移动到GPU上
                            X, y = X.to(device), y.to(device)
                            y_pred = model(X)
                            loss = criterion(y_pred, y.long())
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                            running_loss += loss.item()
                            _, predicted = torch.max(y_pred.data, 1)
                            total += y.size(0)
                            correct += (predicted == y).sum().item()
                        progress.update(task1, advance=1)
                        acc = correct/total
                        losses.append(running_loss)
                        accuracies.append(acc)
                        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_data)}, Accuracy: {acc}')
                
                # 定义文件名
                now = datetime.datetime.now()
                filename = f'Epochs_{EPOCHS}_Batch_size_{BATCH_SIZE}_Learning_rate_{LEARNING_RATE}_Weight_decay_{WEIGHT_DECAY}.bin'

                # 保存模型
                if not os.path.exists("ckpt"):
                    os.makedirs("ckpt")
                torch.save(model.state_dict(), f"ckpt/{filename}")
                
                
                # 测试模型
                model.eval()
                correct = 0
                total = 0
                y_true = []
                y_pred_list = []
                with torch.no_grad():
                    for X, y in DataLoader(dataset=test_data, batch_size=BATCH_SIZE):
                        # 将数据移动到GPU上
                        X, y = X.to(device), y.to(device)
                        y_pred = model(X)
                        _, predicted = torch.max(y_pred.data, 1)
                        total += y.size(0)
                        correct += (predicted == y).sum().item()
                        y_true.extend(y.cpu().tolist())
                        y_pred_list.extend(predicted.cpu().tolist())

                accuracy = correct / total
                f1 = f1_score(y_true, y_pred_list, average='weighted')

                print(f'Epochs: {EPOCHS}, Batch size: {BATCH_SIZE}, Learning rate: {LEARNING_RATE}, Weight decay:, {WEIGHT_DECAY}, Accuracy: {accuracy}, F1 Score: {f1}')
                with open('results.txt', 'a') as f:
                    f.write(f'Epochs: {EPOCHS}, Batch size: {BATCH_SIZE}, Learning rate: {LEARNING_RATE}, Weight decay:, {WEIGHT_DECAY}, Accuracy: {accuracy}, F1 Score: {f1}\n')