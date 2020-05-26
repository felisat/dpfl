import os
import random
import torch
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

def train_op(model, loader, optimizer, epochs=1):
    model.train()  
    for ep in range(epochs):
        running_loss, samples = 0.0, 0
        for x, y in loader:   
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            loss = torch.nn.CrossEntropyLoss()(model(x), y)
            running_loss += loss.item()*y.shape[0]
            samples += y.shape[0]

            loss.backward()
            optimizer.step()  

    return {"loss" : running_loss / samples}
      

def eval_op(model, loader):
    model.train()
    samples, correct = 0, 0

    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            y_ = model(x)
            _, predicted = torch.max(y_.data, 1)
            
            samples += y.shape[0]
            correct += (predicted == y).sum().item()

    return {"accuracy" : correct/samples}


def copy(target, source):
    for name in target:
        target[name].data = source[name].data.clone()
      
def subtract(target, minuend, subtrahend):
    for name in target:
        target[name].data = minuend[name].data.clone()-subtrahend[name].data.clone()
    
def compress(target, compression_fn):
    for name in target:
        target[name].data = compression_fn(target[name].data.clone())
    
def compress_and_accumulate(target, residual, compression_fn):
    for name in target:
        residual[name].data += target[name].data.clone()
        target[name].data = compression_fn(residual[name].data.clone())
        residual[name].data -= target[name].data.clone()
    
def reduce_add_average(target, sources):
    for name in target:
        tmp = torch.mean(torch.stack([source[name].data for source in sources]), dim=0).clone()
        target[name].data += tmp


class FederatedTrainingDevice(object):
    def __init__(self, model_fn, data, batch_size):
        self.model = model_fn().to(device)
        self.data = data
        self.loader = DataLoader(self.data, batch_size=batch_size, shuffle=True)

        self.W = {key : value for key, value in self.model.named_parameters()}
   

    def evaluate(self, loader=None):
        return eval_op(self.model, self.loader if not loader else loader)
  
  
class Client(FederatedTrainingDevice):
    def __init__(self, model_fn, optimizer_fn, data, batch_size, idnum):
        super().__init__(model_fn, data, batch_size)  
        self.optimizer = optimizer_fn(self.model.parameters())
        self.id = idnum
        self.W_old = {key : torch.zeros_like(value) for key, value in self.model.named_parameters()}
        self.dW = {key : torch.zeros_like(value) for key, value in self.model.named_parameters()}
        self.R = {key : torch.zeros_like(value) for key, value in self.model.named_parameters()}
    
    def synchronize_with_server(self, server):
        copy(target=self.W, source=server.W)

    def compute_weight_update(self, epochs=1, loader=None):
        copy(target=self.W_old, source=self.W)
        train_stats = train_op(self.model, self.loader if not loader else loader, self.optimizer, epochs)
        subtract(target=self.dW, minuend=self.W, subtrahend=self.W_old)
        return train_stats   

    def compress_weight_update(self, compression_fn=None, accumulate=False):
        if compression_fn:
            if accumulate:
                compress_and_accumulate(self.dW, self.R, compression_fn)
            else:
                compress(self.dW, compression_fn)



class Server(FederatedTrainingDevice):
    def __init__(self, model_fn, data):
        super().__init__(model_fn, data, batch_size=100)
    
    def select_clients(self, clients, frac=1.0):
        return random.sample(clients, int(len(clients)*frac)) 

    def aggregate_weight_updates(self, clients):
        reduce_add_average(target=self.W, sources=[client.dW for client in clients])


    def save_model(self, path=None, name=None, verbose=True):
        if name:
            if not os.path.exists(path):
                os.makedirs(path)
            torch.save(self.model.state_dict(), os.path.join(path,name))
            if verbose: print("Saved model to", os.path.join(path,name))

    def load_model(self, path=None, name=None, verbose=True):
        if name:
            self.model.load_state_dict(torch.load(os.path.join(path,name)))
            if verbose: print("Loaded model from", os.path.join(path,name))