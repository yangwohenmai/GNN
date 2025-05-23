# https://zhuanlan.zhihu.com/p/383696667
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm
import torch
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data
import numpy as np
import pandas as pd


# 自定义数据集类YooChooseBinaryDataset，用于处理购买预测任务的数据集
# InMemoryDataset类，这是 PyTorch Geometric 中用于处理图数据的基类。
class YooChooseBinaryDataset(InMemoryDataset):
    """
    参数说明：
        - root:数据集的根目录，通常是数据文件存放的文件夹。
        - transform:数据转换函数，用于对数据进行预处理和增强。
        在这个代码段中，没有使用自定义的转换函数，所以传入了默认值 None。
        - pre_transform：预处理转换函数，通常用于在数据加载之前进行一些预处理操作。同样，这里传入了默认值 None。
    """
    def __init__(self, root, df, transform=None, pre_transform=None):
        super(YooChooseBinaryDataset, self).__init__(root, transform, pre_transform)  # transform就是数据增强，对每一个数据都执行
        # 加载已经处理好的数据集。self.processed_paths 存储了已处理数据的文件路径列表，通常情况下只有一个文件。
        # 其中self.data包含了数据对象，self.slices包含了数据对象的切片信息。
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.df = df

    @property  # 这是Python中的装饰器，用于定义属性的getter方法。
    def raw_file_names(self):  # 检查self.raw_dir目录下是否存在raw_file_names()属性方法返回的每个文件
        # 如有文件不存在，则调用download()方法执行原始文件下载
        return []

    # 该方法返回已处理数据文件的文件名列表。
    @property
    def processed_file_names(self):  # 检查self.processed_dir目录下是否存在self.processed_file_names属性方法返回的所有文件，没有就会走process
        return ['yoochoose_click_binary_1M_sess.dataset']

    def download(self):
        pass

    # 用于处理原始数据并将其转化为图数据。
    def process(self):
        # 存储图数据对象
        data_list = []

        # 根据session_id列将原始数据df分成多个会话组。每个会话代表一个用户的一系列行为。
        grouped = self.df.groupby('session_id')
        # session_id存储了当前会话的唯一标识，而group是包含了该会话的所有行的DataFrame。
        for session_id, group in tqdm(grouped):
            # 这一行代码使用LabelEncoder()对会话中的商品ID（item_id）进行标签编码。
            # 标签编码的目的是将原始的商品ID转换为整数形式，以便在图数据中使用。
            sess_item_id = LabelEncoder().fit_transform(group.item_id)
            # 这里将group DataFrame重新索引并删除之前的索引，以确保行索引从零开始并连续。
            group = group.reset_index(drop=True)
            # 在DataFrame group中添加了一个新的列'sess_item_id'，该列包含了标签编码后的商品ID。
            group['sess_item_id'] = sess_item_id
            # 这一行代码用于创建节点特征node_features。具体操作包括：
            # - 从group DataFrame中选择session_id与当前迭代的绘画相同的行。
            # - 选择sess_item_id和item_id这两列数据，并按sess_item_id进行排序
            # - 使用drop_duplicates()方法去除重复的item_id值，并将结果转换为一个Numpy数组。
            node_features = group.loc[group.session_id == session_id, ['sess_item_id', 'item_id']].sort_values('sess_item_id').item_id.drop_duplicates().values
            # 将node_features 转换为PyTorch的LongTensor类型，并添加一个额外的维度，以满足图神经网络的输入要求。
            node_features = torch.LongTensor(node_features).unsqueeze(1)
            # 这两行代码用于创建目标节点和源节点。目标节点是会话中的商品ID（'sess_item_id'）的后续节点，
            # 而源节点是其前一个节点。这是为了构建图数据中的边缘索引。
            target_nodes = group.sess_item_id.values[1:]
            source_nodes = group.sess_item_id.values[:-1]
            # 创建了边缘索引edge_index，其中source_nodes是源节点的列表，target_nodes是目标节点的列表。
            # 这一行代码将它们合并为一个包含两行的 PyTorch LongTensor。
            edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
            # 将节点特征node_features赋给x，表示图数据中的节点特征。
            x = node_features
            # 创建了一个包含购买标签的张量y，这里假设每个会话的购买标签是相同的（取第一个商品的标签作为会话的标签）。
            y = torch.FloatTensor([group.label.values[0]])

            data = Data(x=x, edge_index=edge_index, y=y)
            # 将当前会话的图数据对象添加到data_list列表中，以便后续将它们合并成一个大的数据对象。
            data_list.append(data)
        # 使用pytorch提供的collate方法将图数据对象列表data_list合并成一个数据对象data和一个切片对象slices。
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

def LoadData():
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

    sampled_session_id = np.random.choice(df.session_id.unique(), 1000, replace=False)
    df = df.loc[df.session_id.isin(sampled_session_id)]
    df.nunique()

    df['label'] = df.session_id.isin(buy_df.session_id)
    df.head()
    return df

def GetTrainData():
    df = LoadData()
    dataset = YooChooseBinaryDataset(root='data/', df=df)
    return dataset, df.item_id.max()