import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import os
import time

# 设置随机种子，保证结果可复现
torch.manual_seed(42)

# 数据预处理：转换为Tensor并标准化
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST数据集的均值和标准差
])

# 加载MNIST数据集
train_dataset = datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root='./data', train=False, download=True, transform=transform
)

# 创建数据加载器
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定义MLP模型
class MLP(nn.Module):
    def __init__(self, input_size=784, hidden_sizes=[256, 128], output_size=10):
        super(MLP, self).__init__()
        layers = []
        # 输入层到第一个隐藏层
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.2))
        
        # 隐藏层之间的连接
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
        
        # 输出层
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        # 将二维图像展平为一维向量 (28x28 -> 784)
        x = x.view(x.size(0), -1)
        return self.model(x)

# 初始化模型、损失函数和优化器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 设置TensorBoard
log_dir = os.path.join("MNSIT/runs", f"mnist_mlp_{time.strftime('%Y%m%d_%H%M%S')}")
writer = SummaryWriter(log_dir)

# 向TensorBoard中添加模型图
dummy_input = torch.randn(1, 1, 28, 28).to(device)
writer.add_graph(model, dummy_input)

# 添加一些训练数据到TensorBoard
dataiter = iter(train_loader)
images, labels = next(dataiter)
images = images[:16]  # 取前16张图片
writer.add_images('mnist_samples', images, 0)

# 训练模型
def train(model, train_loader, criterion, optimizer, epoch, writer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播、计算损失、反向传播、参数更新
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # 计算准确率
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
        # 每100个batch记录一次损失
        if batch_idx % 100 == 99:
            step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('training_loss', loss.item(), step)
    
    # 计算epoch的平均损失和准确率
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    writer.add_scalar('epoch_training_loss', epoch_loss, epoch)
    writer.add_scalar('epoch_training_accuracy', epoch_acc, epoch)
    
    print(f'Epoch [{epoch+1}] Training Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
    return epoch_loss, epoch_acc

# 测试模型
def test(model, test_loader, criterion, epoch, writer):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():  # 测试时不计算梯度
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    # 计算测试集的平均损失和准确率
    test_loss /= len(test_loader)
    test_acc = 100 * correct / total
    
    writer.add_scalar('test_loss', test_loss, epoch)
    writer.add_scalar('test_accuracy', test_acc, epoch)
    
    print(f'Test Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%')
    return test_loss, test_acc

# 训练和测试循环
num_epochs = 10
for epoch in range(num_epochs):
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, epoch, writer)
    test_loss, test_acc = test(model, test_loader, criterion, epoch, writer)
    
    # 记录学习率
    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

# 添加模型参数分布到TensorBoard
for name, param in model.named_parameters():
    writer.add_histogram(name, param, num_epochs)
    writer.add_histogram(f'{name}.grad', param.grad, num_epochs)

# 保存模型
path = './MNIST/Model'
if not os.path.exists(path):
    os.makedirs(path)
model_save_path = os.path.join(path, 'mnist_mlp_model.pth')
print(f"Saving model to {model_save_path}")
# 保存模型参数
torch.save(model.state_dict(), model_save_path)

# 关闭TensorBoard写入器
writer.close()

print("\nTraining complete!")
print(f"Model saved as '{model_save_path}'")
print(f"TensorBoard logs saved to: {log_dir}")
print("To view TensorBoard, run: tensorboard --logdir=runs")
