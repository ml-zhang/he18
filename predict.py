from constant_name import *
import torch
import numpy as np
from data_load import Im_data,DataLoader
# from module import module_f1,module_f2,module_f3,module_f4,module_f5
# from module_all_gene import module_
import torch.nn
import torch.optim
from pandas import DataFrame

def pre_dict(s_model,pre_data,save_name=None):
    if type(s_model) == str:
        print('pre_notice')
        model = module_f1()
        if torch.cuda.is_available():
            print('using cuda')
            model = model.cuda()
        model.load_state_dict(torch.load(s_model))
        model.eval()
    else:
        model = s_model


    temp = []
    prob = []
    for cell,_ in pre_data:
        if torch.cuda.is_available():
            cell = cell.cuda()
        out = model(cell)
        _, pred = torch.max(out, 1)
        pre = pred.cpu()
        pd = np.array(pre)
        pd = list(pd)
        temp.append(pd)

    # o_p = save_name
    t=[]
    for item in temp:
        t.extend(item)
    c = 0
    for it in t:
        if it == 0:
            c+=1
    print('the NUM CGE:  ', c,'!!!!!!!!!!!!')
    fi = DataFrame(t)
    fi.index = [i for i in range(fi.shape[0])]
    fi.to_csv(save_name)
    return c

    # with open(o_p,'w',encoding='utf-8') as fi:
    #     for i in range(len(t)):
    #         fi.write(str(t[i])+','+'\n')
    #     fi.close()
def select(tag,s_model):
    if tag == 0:
        print('pre_notice')
        model = module_f1()
        if torch.cuda.is_available():
            print('using cuda')
            model = model.cuda()
        model.load_state_dict(torch.load(s_model))
        model.eval()
    elif tag == 1:
        print('pre_notice')
        model = module_f2()
        if torch.cuda.is_available():
            print('using cuda')
            model = model.cuda()
        model.load_state_dict(torch.load(s_model))
        model.eval()
    elif tag == 2:
        print('pre_notice')
        model = module_f3()
        if torch.cuda.is_available():
            print('using cuda')
            model = model.cuda()
        model.load_state_dict(torch.load(s_model))
        model.eval()
    elif tag == 3:
        print('pre_notice')
        model = module_f4()
        if torch.cuda.is_available():
            print('using cuda')
            model = model.cuda()
        model.load_state_dict(torch.load(s_model))
        model.eval()
    else:
        print('pre_notice')
        model = module_f5()
        if torch.cuda.is_available():
            print('using cuda')
            model = model.cuda()
        model.load_state_dict(torch.load(s_model))
        model.eval()
    return model
if __name__ == '__main__':
    pre_batch_size = 1500
    pre_path = r'D:\PROJECT\mature_to_imm\data\pre_data.csv'
    pre_data_l = Im_data(in_path=pre_path)
    pre_data = DataLoader(pre_data_l, batch_size=pre_batch_size, shuffle=False)
    sl = [
        ('105358.pkl','105358.csv'),
        ('105639.pkl','105639.csv'),
        ('105854.pkl','105854.csv'),
        ('110159.pkl','110159.csv'),
        ('110440.pkl','110440.csv')
    ]
    for i in range(len(sl)):
        smodel = r'model/allgene/'+ sl[i][0]
        s_model = select(i,smodel)
        pre_dict(s_model=s_model, pre_data=pre_data, save_name=sl[i][1])
    #输入数据
    # smodel = r'D:\PROJECT\md\model\allgene\105358.pkl'
    # # # smode2 = r'C:\Users\ZML15\Desktop\mature_to_imm\out_p\05-28_15_44P2.pkl'
    # pre_batch_size = 1500
    # pre_path = r'D:\PROJECT\mature_to_imm\data\pre_data.csv'
    # pre_data_l = Im_data(in_path=pre_path)
    # pre_data = DataLoader(pre_data_l, batch_size=pre_batch_size, shuffle=False)
    # pre_dict(s_model=smodel,pre_data=pre_data,save_name='6354.csv')
    # # pre_dict(s_model=smode2,pre_data=pre_data,save_name='p2')

    # #时间分数据测试
    # smodel = 'out_p/06-03_13_27v1.pkl'
    # # data_name = ['9pre.csv','10pre.csv','13pre.csv','16pre.csv','18pre.csv',]
    # data_name = ['6_pre.csv']
    # pre_batch_size = 100
    # dir_name = 'f_data/'
    # for i in range(len(data_name)):
    #     ip = dir_name+data_name[i]
    #     save_name = 'out_p/'+data_name[i]
    #     pre_data_l = Im_data(in_path=ip)
    #     pre_data = DataLoader(pre_data_l,batch_size=pre_batch_size,shuffle=False)
    #     pre_dict(s_model=smodel,pre_data=pre_data,save_name=save_name)
