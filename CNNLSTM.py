
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
from torch import from_numpy
from sklearn import metrics
import numpy as np

def Predict(model, X):
  try:
    return model(from_numpy(X).type(torch.float).to('cuda'))
  except:
    data_set = TestDataset(X)
    pred = torch.empty((1,5)).type(torch.float).cpu()
    data_loader = DataLoader(data_set, batch_size = 512)
    for x in data_loader:
      pred = torch.vstack((pred, model(x.type(torch.float).cuda()).cpu()))
    return pred[1:]

def Train (num_epochs, model0, model1, loaders, loss_func, lr, X_train_sequence, Y_train, wd = 0):
  model0.train()
  total_step = len(loaders)
  y_label = np.argmax(Y_train, axis = 1)
  
  optimizer1 = optim.Adam(model1.parameters(), lr = 2*lr, weight_decay = wd)

  precision0 , precision1 = 0, 0

  for epoch in range(num_epochs):
    for i, (x, y) in enumerate(loaders):

      x, y = x.type(torch.float).to('cuda'), y.type(torch.float).to('cuda')
            
      out0 = model0(x)
      out1 = model1(x)
      loss0 = loss_func(out0, y)
      loss1 = loss_func(out1, y)

      optimizer0 = optim.Adam(model0.parameters(), lr = lr/(2**(epoch//10)), weight_decay = wd)

      if epoch < 11: 
        optimizer1.param_groups[0]['lr'] = 2*lr/( (epoch//3 + 1)*2 )
      else:
        optimizer1.param_groups[0]['lr'] = 2*lr/(2*4**(epoch//10 + 2))
            
      optimizer0.zero_grad()
      optimizer1.zero_grad()

      loss0.backward()
      optimizer0.step()

      loss1.backward()
      optimizer1.step()

      if (i+1) % total_step == 0:
        y_pred0 = torch.argmax(Predict(model0, X_train_sequence), dim = 1, keepdim= True).cpu()
        y_pred1 = torch.argmax(Predict(model1, X_train_sequence), dim = 1, keepdim= True).cpu()

        precision0 = metrics.precision_score(y_label, y_pred0, average = None)
        precision1 = metrics.precision_score(y_label, y_pred1, average = None)

        y_pred_merge = torch.zeros_like(y_label)
        for i in range (len(y_label)):
          if precision0[y_pred0[i]] > precision1[y_pred1[i]]: 
            y_pred_merge[i] = y_pred0[i]
          else:
            y_pred_merge[i] = y_pred1[i]

        print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, f1: {:.4f}' 
                       .format(epoch + 1, num_epochs, i + 1, total_step, loss0.item(),metrics.f1_score(y_label, y_pred_merge, average='macro')))

  return precision0, precision1


class TrainDataset(Dataset):
    def __init__(self, text_data, text_label):
        super().__init__()
        self.size = text_data.shape[0]
        self.data = torch.from_numpy(np.double(text_data))
        self.label = torch.from_numpy(np.double(text_label))
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        data_point = self.data[idx]
        data_label = self.label[idx]
        return data_point, data_label
class TestDataset(Dataset):
    def __init__(self, text_data):
        super().__init__()
        #text_data is a np.ndarray
        self.size = text_data.shape[0]
        self.data = torch.from_numpy(np.double(text_data))
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        data_point = self.data[idx]

        return data_point
class f1_loss(nn.Module):
  def __init__(self, weight = None):
    super(f1_loss, self).__init__()
    self.weight = weight


  def forward(self, y_pred, y_true):

    tp = torch.sum(y_true*y_pred, dim=0)
    
    fp = torch.sum((1-y_true)*y_pred, dim=0)
    fn = torch.sum(y_true*(1-y_pred), dim=0)

    p = tp / (tp + fp + 1e-10)
    r = tp / (tp + fn + 1e-10)

    f1 = 2*p*r / (p+r+1e-10)
    f1 = torch.where(torch.isnan(f1), torch.zeros_like(f1), f1)
    if self.weight.shape[0] != 0:
      f1 = f1 * torch.Tensor(self.weight).cuda()

    return 1 - torch.mean(f1)
class CNNLSTM(nn.Module):
    def __init__(self, num_classes, input_size, num_layers):
        super(CNNLSTM, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels = input_size, out_channels = input_size, kernel_size = 3, padding = 1, stride = 1),
            nn.ReLU(),
            nn.BatchNorm1d(input_size),
            nn.MaxPool1d(2),
          
        )

        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size = 50,
                          num_layers=num_layers, batch_first=True, dropout = 0.2) #CNNLSTM

        self.lstm2 =  nn.LSTM(input_size = 50, hidden_size = 50, batch_first = True)
        self.fc1 = nn.Linear(50, num_classes) #fully connected last layer
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
    
    def forward(self, x):
        inp = torch.moveaxis(x, 1, 2)
        inp = self.cnn(inp)
        inp = torch.moveaxis(inp, 1, 2)

        self.out1, _ = self.lstm1(inp) #CNNLSTM with input, hidden, and internal state
        self.out2, (hn2, _) = self.lstm2(self.out1)

        hn2 = hn2.view(-1, 50) #reshaping the data for Dense layer next
        out = self.relu(hn2)
        out = self.fc1(out) #first Dense
        out = self.softmax(out) #relu
        return out 