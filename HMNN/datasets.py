import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import numpy as np




def get_train_test_data2(all_data,  k, test_ratio=0.2,seed=42,eps="random"):
    x_train_list=[]
    y_train_list=[]
    x_test_list=[]
    y_test_list=[]
    for i in range(len(all_data)):
        X, y = all_data[i][:, :19], all_data[i][:, -4:]
        if eps == "random":
            X[:,-1] = np.random.random(len(all_data[i]))
        elif eps == "error":
            X[:, -1] = 0
        else:
            pass
        np.random.shuffle(y)
        if i!=k:
            x_train_list.append(X)
            y_train_list.append(y)
        if i==k:
            x_test_list.append(X)
            y_test_list.append(y)
    x_train, x_test, y_train, y_test = np.concatenate(x_train_list),np.concatenate(x_test_list),np.concatenate(y_train_list),np.concatenate(y_test_list)
    print(f"the number of train samples is: {len(x_train)}")
    print(f"the number of test samples is: {len(x_test)}")
    return x_train, x_test, y_train, y_test


def get_train_test_data1(all_data, test_ratio=0.2,seed=42,eps="random"):
    x_train_list=[]
    y_train_list=[]
    x_test_list=[]
    y_test_list=[]
    for i in range(len(all_data)):
        X, y = all_data[i][:, :19], all_data[i][:, -4:]
        if eps == "random":
            X[:,-1] = np.random.random(len(all_data[i]))
        elif eps == "error":
            X[:, -1] = 0
        else:
            pass
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio,random_state=seed)
        x_train_list.append(x_train)
        y_train_list.append(y_train)
        x_test_list.append(x_test)
        y_test_list.append(y_test)
    x_train, x_test, y_train, y_test = np.concatenate(x_train_list),np.concatenate(x_test_list),np.concatenate(y_train_list),np.concatenate(y_test_list)
    print(f"the number of train samples is: {len(x_train)}")
    print(f"the number of test samples is: {len(x_test)}")
    return x_train, x_test, y_train, y_test



class MMoE_Dataset(Dataset):
    def __init__(self, x, y1, y2, y3, y4):
        self.y1 = torch.FloatTensor(y1)
        self.y2 = torch.FloatTensor(y2)
        self.y3 = torch.FloatTensor(y3)
        self.y4 = torch.FloatTensor(y4)
        self.x = torch.FloatTensor(x)
    def __getitem__(self, idx):
        return self.x[idx], self.y1[idx], self.y2[idx], self.y3[idx], self.y4[idx]
    def __len__(self):
        return len(self.x)

def get_train_test_data(file_path,  train_index , test_index,eps="random"):
    data = pd.read_csv(file_path).values
    X, y = data[:, :19], data[:, -4:]
    if eps == "random":
        X[:,-1] = np.random.random(len(data))
    elif eps == "error":
        X[:, -1] = 0
    else:
        pass
    x_train, x_test, y_train, y_test = X[train_index],X[test_index],y[train_index],y[test_index]
    print(f"the number of train samples is: {len(x_train)}")
    print(f"the number of test samples is: {len(x_test)}")
    return x_train, x_test, y_train, y_test

