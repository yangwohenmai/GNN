import torch
import numpy as np
from torch_geometric.loader import DataLoader
from  sklearn.metrics import roc_auc_score
import MyModel
import InMemoryDataset模板




def train():
    model.train()
 
    loss_all = 0
    for data in train_loader:
        data = data
        #print('data',data)
        optimizer.zero_grad()
        output = model(data)
        label = data.y
        loss = crit(output, label)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len(dataset)

def evalute(loader,model):
    model.eval()

    prediction = []
    labels = []

    with torch.no_grad():
        for data in loader:
            data = data#.to(device)
            pred = model(data)#.detach().cpu().numpy()

            label = data.y#.detach().cpu().numpy()
            prediction.append(pred)
            labels.append(label)
    prediction =  np.hstack(prediction)
    labels = np.hstack(labels)

    return roc_auc_score(labels,prediction) 

    

# 主函数
if __name__ == '__main__':
    dataset, num_embeddings = InMemoryDataset模板.GetTrainData()
    model = MyModel.Net(num_embeddings+1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    crit = torch.nn.BCELoss()
    train_loader = DataLoader(dataset, batch_size=64)
    for epoch in range(10):
        print('epoch:',epoch)
        loss = train()
        print(loss)

    for epoch in range(1):
        roc_auc_score = evalute(dataset,model)
        print('roc_auc_score',roc_auc_score)
