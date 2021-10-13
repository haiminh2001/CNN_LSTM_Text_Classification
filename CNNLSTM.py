
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
def Train (num_epochs, model, loaders, loss_func, lr, X_train_sequence, Y_train, wd = 0):
  model.train()
  total_step = len(loaders)
  
  for epoch in range(num_epochs):
    for i, (x, y) in enumerate(loaders):

      x, y = x.type(torch.float).to('cuda'), y.type(torch.float).to('cuda')
            
      out = model(x)
      loss = loss_func(out, y)

            
      optimizer = optim.Adam(model.parameters(), lr = lr/(2**(epoch//10)), weight_decay = wd)
      optimizer.zero_grad()

      loss.backward()
      optimizer.step()

      if (i+1) % total_step == 0:
        Y_pred = Predict(model, X_train_sequence)
        check1 = torch.argmax(Y_pred, dim = 1, keepdim= True).cpu()
        ytrain1 = np.argmax(Y_train, axis = 1)
        print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, f1: {:.4f}' 
                       .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),metrics.f1_score(ytrain1, check1, average='macro')))
        del(Y_pred)

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
            nn.Dropout(0.2),
            # nn.Conv1d(in_channels = vector_size, out_channels = vector_size, kernel_size = 3, padding = 1, stride = 1),
            # nn.MaxPool1d(2)
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