import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader,TensorDataset
from torch.autograd import Variable
import numpy as np
import random as rd

class man_made_da(Dataset):
    def __init__(self):
        self.len = 150
        self.x = torch.randn(self.len,15949)
        self.y = [rd.randint(0,1) for i in range(self.len)]
        pass
    def __getitem__(self, index):
        return self.x[index],self.y[index]
    def __len__(self):
        return self.len

class Im_data(Dataset):
    '''

    '''
    def __init__(self,in_path,lab_path = ''):
        self.in_path = in_path
        self.lab_path = lab_path
        self.load()
    def load(self):
        data = pd.read_csv(self.in_path, index_col=0,dtype=np.float32)
        self.len = data.shape[0]
        data = data.values
        self.input_data = torch.from_numpy(data)
        if self.lab_path == '':
            self.label = torch.zeros(data.shape[0],dtype=torch.float32)
        else:
            label = pd.read_csv(self.lab_path,index_col=0,header=0).values.T
            label = label[0]
            self.cell_names,self.label = np.unique(label,return_inverse=True)
            self.label = torch.tensor(self.label)
            # self.label = torch.IntTensor(self.label)
            #label 按0,1,2……与cell_name[index]一一对应
    def __getitem__(self, index):
        return self.input_data[index],self.label[index]
    def __len__(self):
        return self.len

# class DataLoader_sep(object):
#     super(DataLoader).__init__()
#     def __init__(self,bs):
#         self.bs = bs








if __name__ == "__main__":
    print('start')
    #当被当成模块引入，则不会执行，方便调试的不用理会
    #测试数据载入接口
    t_path = 'data/train_ml.csv'
    label_path = 'data/label_ml.csv'
    pre_path =  'data/predict_data_t.csv'
    small_path = 'data/val_data.csv'
    # txt_path = 'data/scale_data.txt'
    test_withlabel = Im_data(in_path=small_path,lab_path=label_path)
    test_no_label = Im_data(in_path=pre_path)

    d1,l1 = test_withlabel[1]
    d2,l2 = test_no_label[999]
    #
    print(d1[:10])
    print(l1)
    print(len(d2))
    print(l2)

    # test_txt = Im_data(txt_path)
    print('stop_point')
