
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Subset


def get_data(name, **kw_args):
    if name=="EMNIST": return get_EMNIST(**kw_args)

def get_EMNIST(n_clients, alpha, n_train=50000, n_test=10000):
    data = datasets.EMNIST(root=".", split="byclass", download=True)
    labels = data.targets.numpy()

    idcs = np.random.permutation(len(data))
    client_idcs, server_idcs = idcs[:n_train], idcs[n_train:n_train+n_test]
    
    client_subset_idcs = split_noniid(client_idcs, n_clients, labels, alpha)

    client_data = [CustomSubset(data, idcs, transforms.ToTensor()) for idcs in client_subset_idcs]
    server_data = CustomSubset(data, server_idcs, transforms.ToTensor())

    return client_data, server_data


def split_noniid(train_idcs, n_clients, train_labels, alpha):
    '''Splits data among the clients according to a dirichlet distribution with parameter alpha'''
    n_classes = np.max(train_labels)+1
    label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)

    class_idcs = [np.argwhere(train_labels[train_idcs]==y).flatten() 
           for y in range(n_classes)]

    client_idcs = [[] for _ in range(n_clients)]
    for c, fracs in zip(class_idcs, label_distribution):
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1]*len(c)).astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [train_idcs[np.concatenate(idcs)] for idcs in client_idcs]

    print_split(client_idcs, train_labels)
  
    return client_idcs


class CustomSubset(Subset):
    def __init__(self, dataset, indices, subset_transform=None):
        super().__init__(dataset, indices)
        self.subset_transform = subset_transform
        
    def __getitem__(self, idx):
        x, y = self.dataset[self.indices[idx]]
        if self.subset_transform:
            x = self.subset_transform(x)
      
        return x, y   


def print_split(idcs, labels):
    n_labels = np.max(labels) + 1 
    print("Data split:")
    for i, idccs in enumerate(idcs):
        if i < 10 or i>len(idcs)-10:
            split = np.sum(np.array(labels)[idccs].reshape(1,-1)==np.arange(n_labels).reshape(-1,1), axis=1)
            print(" - Client {}: {}".format(i,split), flush=True)
        elif i==len(idcs)-10:
            print(".  "*10+"\n"+".  "*10+"\n"+".  "*10)
    print()