import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import string
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from loguru import logger
import time

# 1. 数据预处理：将名称转换为特征
class NameDataset(Dataset):
    def __init__(self, names, genders, max_len=10):
        self.names = names
        self.genders = genders
        self.max_len = max_len

        # names to characters and create a character set
        self.names = [name.lower() for name in names]  # 转为小写
        # self.name_chars = set(''.join(self.names))
        self.chars = set(''.join(self.names))
        self.char_to_idx = {c: i+1 for i, c in enumerate(self.chars)}  # 0留作填充
        self.vocab_size = len(self.chars) + 1

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx].lower()
        # 字符编码：将名称转换为固定长度的索引序列
        name_encoded = [self.char_to_idx.get(c, 0) for c in name[:self.max_len]]
        # 填充到固定长度
        name_encoded += [0] * (self.max_len - len(name_encoded))
        # 性别编码：0->女，1->男
        gender = 1 if self.genders[idx] == 'M' else 0
        return (torch.tensor(name_encoded, dtype=torch.long),
                torch.tensor(gender, dtype=torch.float32))

# 2. 模型设计：带嵌入层的MLP
class NameGenderMLP(nn.Module):
    def __init__(self, vocab_size, embed_dim=16, hidden_dim=64, max_len=10):
        super().__init__()
        self.max_len = max_len
        # 字符嵌入层：将字符索引转换为向量
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # MLP层
        self.fc1 = nn.Linear(embed_dim * max_len, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)  # 二分类输出
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.3)  # 防止过拟合

    def forward(self, x):
        # x shape: (batch_size, max_len)
        x = self.embedding(x)  # (batch_size, max_len, embed_dim)
        x = x.view(x.size(0), -1)  # 展平为(batch_size, max_len*embed_dim)
        x = self.relu1(self.fc1(x))
        x = self.dropout(x)
        x = self.relu2(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # 输出0-1之间的概率
        return x

# 3. 训练函数
def train_model(model, train_loader, val_loader, epochs=50, lr=0.001):
    logger.info(f'Start training with MLP model, parameters: {model}, epochs: {epochs}, lr: {lr}')
    
    # Setup tensorboard
    from torch.utils.tensorboard import SummaryWriter
    log_dir = f'name2gender/runs/name_gender_mlp_{time.strftime("%Y%m%d_%H%M%S")}'
    writer = SummaryWriter(log_dir)
    logger.info(f'TensorBoard log directory: {log_dir}')
    # 记录模型图
    dummy_input = torch.randint(0, model.embedding.num_embeddings, (1, model.max_len)).to(next(model.parameters()).device)
    writer.add_graph(model, dummy_input)
    
    criterion = nn.BCELoss()  # 二分类交叉熵损失
    optimizer = optim.Adam(model.parameters(), lr=lr)  # Adam优化器通常效果更好
    
    for epoch in range(epochs):
        logger.info(f'Starting epoch {epoch+1}/{epochs}')
        # timing
        start_time = time.time()
        model.train()
        train_loss = 0.0
        for names, genders in train_loader:
            names = names.to(next(model.parameters()).device)      # <--- add this
            genders = genders.to(next(model.parameters()).device)  # <--- add this
            optimizer.zero_grad()  # 清零梯度
            outputs = model(names).squeeze()
            loss = criterion(outputs, genders)
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            train_loss += loss.item() * names.size(0)
        
        # 验证
        model.eval()
        val_loss = 0.0
        y_pred = []
        y_true = []
        with torch.no_grad():
            for names, genders in val_loader:
                names = names.to(next(model.parameters()).device)      # <--- add this
                genders = genders.to(next(model.parameters()).device)  # <--- add this
                outputs = model(names).squeeze()
                loss = criterion(outputs, genders)
                val_loss += loss.item() * names.size(0)
                y_pred.extend((outputs > 0.5).float().tolist())
                y_true.extend(genders.tolist())
        
        # 计算指标
        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        val_acc = accuracy_score(y_true, y_pred)
        
        # 记录到TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

        if True or (epoch + 1) % 5 == 0:
            logger.info(f'Epoch {epoch+1}/{epochs}')
            logger.info(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')
            logger.info(f'Time: {time.time() - start_time:.2f}s')

# 4. 示例数据和训练
if __name__ == "__main__":
    # 示例数据（实际应用中应使用更大的数据集）
    names = [
        "Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", 
        "Grace", "Henry", "Ivy", "Jack", "Jill", "Kevin"
    ]
    genders = [
        "female", "male", "male", "female", "female", "male",
        "female", "male", "female", "male", "female", "male"
    ]
    csv_path = 'name2gender/nameGender.Shifenzheng.20050144.csv'
    df = pd.read_csv(csv_path, on_bad_lines='skip')
    # Drop rows which Gender is not M or F
    df = df[df['Gender'].isin(['M', 'F'])] # # M           12773969, F            6479097

    # Get 6479097 M and 6479097 F
    df = df.groupby('Gender').sample(n=6479097, random_state=42)

    # names to str
    df['Name'] = df['Name'].astype(str)
    names = df['Name'].tolist()
    genders = df['Gender'].tolist()


    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # check for mps
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    
    # 划分训练集和验证集
    train_names, val_names, train_genders, val_genders = train_test_split(
        names, genders, test_size=0.2, random_state=42
    )
    
    # 创建数据集和数据加载器
    train_dataset = NameDataset(train_names, train_genders)
    val_dataset = NameDataset(val_names, val_genders)
    train_loader = DataLoader(train_dataset, batch_size=4000, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4000, shuffle=False)
    
    logger.info(f'Using device: {device}')
    # 初始化模型并训练
    model = NameGenderMLP(
        vocab_size=train_dataset.vocab_size,
        embed_dim=16,
        hidden_dim=64,
        max_len=10
    ).to(device)
    
    train_model(model, train_loader, val_loader, epochs=50, lr=0.001)
