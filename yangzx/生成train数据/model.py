import baostock as bs
import 构建图train数据
import 获取股市码表
import 获取个股票行情

from torch_geometric.nn import TopKPooling, SAGEConv
from torch_geometric.nn import global_add_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
import torch
import numpy as np

from sklearn.preprocessing import LabelEncoder
import pandas as pd
#用户的点击行为数据 
df = pd.read_csv('E:\\MyGit\\BigDataFile\\yoochoose-data\\yoochoose-clicks.dat', header=None)
df.columns=['session_id','timestamp','item_id','category']
#用户有没有购买商品 
buy_df = pd.read_csv('E:\\MyGit\\BigDataFile\\yoochoose-data\\yoochoose-buys.dat', header=None)
buy_df.columns=['session_id','timestamp','item_id','price','quantity']
 
item_encoder = LabelEncoder()
df['item_id'] = item_encoder.fit_transform(df.item_id)
 
"""
session_id相同代表是同一个人, 点了四个网页----某一个人的点击行为
item_id:代表东西是什么(商品id号)
"""
df.head()
buy_df.head()

from torch_geometric.data import InMemoryDataset
from tqdm import tqdm
from torch_geometric.data import Data
 
class YooChooseBinaryDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(YooChooseBinaryDataset, self).__init__(root, transform, pre_transform) # transform就是数据增强，对每一个数据都执行
        self.data, self.slices = torch.load(self.processed_paths[0])
 
    @property #python装饰器， 只读属性，方法可以像属性一样访问
    def raw_file_names(self): #①检查self.raw_dir目录下是否存在raw_file_names()属性方法返回的每个文件 
                              #②如有文件不存在，则调用download()方法执行原始文件下载
        return []
    @property
    def processed_file_names(self): #③检查self.processed_dir目录下是否存在self.processed_file_names属性方法返回的所有文件，有则直接加载
                                    #④没有就会走process,得到'yoochoose_click_binary_1M_sess.dataset'文件
        return ['yoochoose_click_binary_1M_sess.dataset']
 
    def download(self):#①检查self.raw_dir目录下是否存在raw_file_names()属性方法返回的每个文件 
                       #②如有文件不存在，则调用download()方法执行原始文件下载
        pass
    
    def process(self):#④没有就会走process,得到'yoochoose_click_binary_1M_sess.dataset'文件
        
        data_list = [] #保存最终生成图的结果
 
        # process by session_id
        grouped = df.groupby('session_id')
        for session_id, group in tqdm(grouped):
            sess_item_id = LabelEncoder().fit_transform(group.item_id)
            group = group.reset_index(drop=True)
            group['sess_item_id'] = sess_item_id
            node_features = group.loc[group.session_id==session_id,['sess_item_id','item_id']].sort_values('sess_item_id').item_id.drop_duplicates().values
 
            node_features = torch.LongTensor(node_features).unsqueeze(1)
            target_nodes = group.sess_item_id.values[1:]
            source_nodes = group.sess_item_id.values[:-1]
 
            edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
            x = node_features
 
            y = torch.FloatTensor([group.label.values[0]])
            #创建图
            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)
        
        data, slices = self.collate(data_list)#转换成可以保存到本地的格式
        torch.save((data, slices), self.processed_paths[0])#保存操作，名字跟yoochoose_click_binary_1M_sess.dataset一致


embed_dim = 128
import numpy as np
#数据有点多，咱们只选择其中一小部分来建模
#unique：唯一性索引
#选择十万条来建模
sampled_session_id = np.random.choice(df.session_id.unique(), 100000, replace=False)
df = df.loc[df.session_id.isin(sampled_session_id)]
df.nunique()

df['label'] = df.session_id.isin(buy_df.session_id)
df.head()

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = SAGEConv(embed_dim, 128) #卷积层 输入embed_dim，输出128
        self.pool1 = TopKPooling(128, ratio=0.8) #做剪枝操作
        self.conv2 = SAGEConv(128, 128)
        self.pool2 = TopKPooling(128, ratio=0.8)
        self.conv3 = SAGEConv(128, 128)
        self.pool3 = TopKPooling(128, ratio=0.8)
        self.item_embedding = torch.nn.Embedding(num_embeddings=df.item_id.max() +10, embedding_dim=embed_dim)#映射向量
        self.lin1 = torch.nn.Linear(128, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, 1)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.act1 = torch.nn.ReLU()
        self.act2 = torch.nn.ReLU()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch # x:n*1,其中每个图里点的个数是不同的
        #print(x)
        x = self.item_embedding(x)# n*1*128 特征编码后的结果
        #print('item_embedding',x.shape)
        x = x.squeeze(1) # n*128        
        #print('squeeze',x.shape)
        
        """
        对输入不断做卷积，不断做池化池化，得到的特征会越来越浓缩，图会越来越小，
        但是池化完成之后的特征维度都是一样的
        
        """
        x = F.relu(self.conv1(x, edge_index))# n*128
        #print('conv1',x.shape)
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)# pool之后得到 n*0.8个点
        #print('self.pool1',x.shape)
        #print('self.pool1',edge_index)
        #print('self.pool1',batch)
        #x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x1 = gap(x, batch)  #   gap:全局平均池化  得到全局特征
        #print('gmp',gmp(x, batch).shape) # batch*128
        #print('cat',x1.shape) # batch*256
        x = F.relu(self.conv2(x, edge_index))
        #print('conv2',x.shape)
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        #print('pool2',x.shape)
        #print('pool2',edge_index)
        #print('pool2',batch)
        #x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x2 = gap(x, batch)
        #print('x2',x2.shape)
        x = F.relu(self.conv3(x, edge_index))
        #print('conv3',x.shape)
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        #print('pool3',x.shape)
        #x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x3 = gap(x, batch)
        #print('x3',x3.shape)# batch * 256
        x = x1 + x2 + x3 # 获取不同尺度的全局特征
        """通过全连接层，得到最终输出结果值"""
        x = self.lin1(x)
        #print('lin1',x.shape)
        x = self.act1(x)
        x = self.lin2(x)
        #print('lin2',x.shape)
        x = self.act2(x)      
        x = F.dropout(x, p=0.5, training=self.training)
 
        x = torch.sigmoid(self.lin3(x)).squeeze(1)#batch个结果
        #print('sigmoid',x.shape)
        return x
    

from torch_geometric.loader import DataLoader


def train():
    model.train()
 
    loss_all = 0
    for data in train_loader:#遍历dataloader
        data = data
        #print('data',data)
        optimizer.zero_grad()
        output = model(data)#data数据传入模型
        label = data.y
        loss = crit(output, label)#计算损失
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()#梯度更新
    return loss_all / len(dataset)

lg = bs.login()
stockPriceDic = 获取个股票行情.GetStockPriceDWMBaostock('600000.SH', 0)
dataset = 构建图train数据.TrainData(stockPriceDic)
model = Net()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
crit = torch.nn.BCELoss()
train_loader = DataLoader(dataset, batch_size=64)
for epoch in range(10):
    print('epoch:',epoch)
    loss = train()
    print(loss)
 
from  sklearn.metrics import roc_auc_score
 
def evalute(loader,model):
    model.eval()
 
    prediction = []
    labels = []
 
    with torch.no_grad():
        for data in loader:
            data = data#.to(device)
            pred = model(data[0])#.detach().cpu().numpy()
 
            label = data.y#.detach().cpu().numpy()
            prediction.append(pred)
            labels.append(label)
    prediction =  np.hstack(prediction)
    labels = np.hstack(labels)
 
    return roc_auc_score(labels,prediction) 
 
 
for epoch in range(1):
    roc_auc_score = evalute(dataset,model)
    print('roc_auc_score',roc_auc_score)