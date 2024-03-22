import torch
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

# 假设我们有一些图数据
# x: 节点特征矩阵 (NumNodes x NumFeatures)
# edge_index: 边的索引矩阵
# y: 节点的标签 (NumNodes)

x = torch.tensor([[0., 1.], [1., 0.], [0., 1.], [1., 0.]])  # 假设有4个节点，每个节点有2个特征
edge_index = torch.tensor([[0, 1, 3, 2], [1, 0, 2, 3]])  # 边的索引，表示节点0和1相连，节点2和3相连
y = torch.tensor([0, 1, 1, 0])  # 节点标签，0和1类

# 创建Data对象
data = Data(x=x, edge_index=edge_index, y=y)

# 定义GCN模型
class GCN(torch.nn.Module):
    def __init__(self, in_features, hidden_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_features, hidden_features)
        self.conv2 = GCNConv(hidden_features, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

# 实例化模型
model = GCN(in_features=x.size(1), hidden_features=16, num_classes=y.max().item() + 1)

# 定义损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    optimizer.zero_grad()  # 清空梯度
    logits = model(data.x, data.edge_index)  # 前向传播
    loss = criterion(logits, data.y)  # 计算损失
    loss.backward()  # 反向传播
    optimizer.step()  # 更新参数

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 测试模型
with torch.no_grad():
    logits = model(data.x, data.edge_index)
    pred = logits.argmax(dim=1)
    correct = pred.eq(data.y).sum().item()
    accuracy = 100 * correct / len(data.y)
    print(f'Accuracy: {accuracy:.2f}%')







