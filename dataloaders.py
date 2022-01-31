import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np


#creating a custom dataset class to reshape the image tensors on sample load, since conv2d requires 4 dimensional input
#could have also been done through collate_fn I guess
class MNIST(Dataset):
    def __init__(self, raw:np.ndarray):
        self.y = raw[:,0]
        self.X = raw[:,1:]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        label = self.y[index]
        dat = self.X[index].reshape(28,28)
        dat = np.expand_dims(dat, axis=0)

        label = torch.tensor(label, dtype=torch.long)
        dat = torch.tensor(dat, dtype=torch.double)

        return label, dat


def load_data_old(hparams):
    #read CSVs into dataframes
    test = pd.read_csv("mnist_test.csv")
    train = pd.read_csv("mnist_train.csv")

    #seperate the target variable
    ytrain = train['label'].copy()
    train.drop('label', axis=1, inplace=True)
    ytest = test['label'].copy()
    test.drop('label', axis=1, inplace=True)

    #load data into pytorch tensors
    xtrain = torch.tensor(train.to_numpy())
    xtest = torch.tensor(test.to_numpy())
    ytrain = torch.tensor(ytrain.to_numpy())
    ytest = torch.tensor(ytest.to_numpy())


def load_data(hparams):
    test = pd.read_csv(hparams.data_path + "mnist_test.csv").to_numpy()
    train = pd.read_csv(hparams.data_path + "mnist_train.csv").to_numpy()

    test = MNIST(test)
    train = MNIST(train)

    trainloader = DataLoader(train, batch_size=hparams.batch_size, num_workers=hparams.num_workers, shuffle=True)

    testloader = DataLoader(test, batch_size=hparams.batch_size, num_workers=hparams.num_workers)

    return trainloader, testloader