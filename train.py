from constant_name import *
import torch
import torch.nn
import torch.optim
from torch.autograd import Variable
# from module import module_f1,module_f2,module_f3,module_f4,module_f5
from module_all_gene import module_hid, module_sim, module_mid, module_expend
from matplotlib import pyplot as plt
import time


# use pretrain model
# def train_model(train_data):
#     #input train_data
#     # train model
#     # return model and the save name
#     model = module_v4()
#     if torch.cuda.is_available():
#         print('using cuda')
#         model = model.cuda()
#
#     criterion = torch.nn.CrossEntropyLoss()
#     optimizer = torch.optim.SGD(model.parameters(), lr=l_rate)
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.8)
#
#     #train
#     ba_id = 0
#     mpt=[]
#     for ep in range(num_epoch):
#         print('epoch_id',ep)
#         mpt1 =[]
#         scheduler.step()
#         print(scheduler.get_lr()[0])
#         for da in train_data:
#             cell, cl = da
#             if torch.cuda.is_available():
#                 cell = cell.cuda()
#                 cl = cl.cuda()
#             else:
#                 cell = Variable(cell)
#                 cl = Variable(cl)
#
#             out = model(cell)
#             loss = criterion(out,cl)
#
#             print_loss = loss.data.item()
#             print_loss = float(print_loss)
#             mpt1.append(print_loss)
#
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             ba_id +=1
#             print('batch_id: {}, loss: {:.4}'.format(ba_id, loss.data.item()))
#         mpt.append(sum(mpt1)/len(mpt1))
#
#     time_tag = time.strftime('%m-%d_%H_%M',time.localtime(time.time()))
#     save_name = out_path + time_tag + 'v1.pkl'
#
#     torch.save(model.state_dict(), save_name)
#
#     x_l = len(mpt)
#     plt.plot([i for i in range(x_l)],mpt)
#     plt.show()
#     plt.savefig(out_path+time_tag+'loss.jpg')
#
#     return model,save_name
#
# def bal_train(*args):
#     tu_train_data = args
#     #input train_data
#     # balance train the model
#     # return model and the save name
#     model = module_v1()
#     if torch.cuda.is_available():
#         print('using cuda')
#         model = model.cuda()
#
#     criterion = torch.nn.CrossEntropyLoss()
#     optimizer = torch.optim.SGD(model.parameters(), lr=l_rate)
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=5,gamma=0.8)
#
#     #train
#     ba_id = 0
#     mpt=[]
#     for ep in range(num_epoch):
#         print('epoch_id',ep)
#         scheduler.step()
#         print(scheduler.get_lr()[0])
#         for train_data in tu_train_data:
#             mpt1 = []
#             for da in train_data:
#                 cell, cl = da
#                 if torch.cuda.is_available():
#                     cell = cell.cuda()
#                     cl = cl.cuda()
#                 else:
#                     cell = Variable(cell)
#                     cl = Variable(cl)
#
#                 out = model(cell)
#                 loss = criterion(out,cl)
#
#                 print_loss = loss.data.item()
#                 print_loss = float(print_loss)
#                 mpt1.append(print_loss)
#
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
#
#                 ba_id +=1
#                 print('batch_id: {}, loss: {:.4}'.format(ba_id, loss.data.item()))
#             mpt.append(sum(mpt1)/len(mpt1))
#
#     time_tag = time.strftime('%m-%d_%H_%M',time.localtime(time.time()))
#     save_name = out_path + time_tag + 'v1.pkl'
#
#     torch.save(model.state_dict(), save_name)
#
#     x_l = len(mpt)
#     plt.plot([i for i in range(x_l)],mpt)
#     plt.show()
#     plt.savefig(out_path+time_tag+'loss.jpg')
#
#     return model,save_name

def init_model(model_tag=1):
    # model = module_version2_v1()
    if model_tag == 1:
        model = module_hid()
        print('with hidden')
    elif model_tag == 2:
        model = module_sim()
        print('simple')
    elif model_tag == 3:
        model = module_mid()
        print('middle')
    else:
        model = module_expend()
        print('expend')
    # model = module_f2()
    # 实例化（）记得加，忘了好几次了
    model.train()
    if torch.cuda.is_available():
        print('using cuda')
        model = model.cuda()
    model.criterion = torch.nn.CrossEntropyLoss()
    model.optimizer = torch.optim.SGD(model.parameters(), lr=l_rate)
    model.scheduler = torch.optim.lr_scheduler.StepLR(model.optimizer, step_size=10, gamma=0.5)
    return model


def train_model_version_two(train_data, epoch, model=None, model_tag=4):
    # 单纯的一次训练过程，所有其他事情交由控制器来做，只返回模型和本次的loss。
    if epoch == 0:
        model = init_model(model_tag)
    else:
        model = model
        model.train()
    model.scheduler.step()
    # scheduler.step()
    print(model.scheduler.get_lr()[0])
    mpt1 = []
    for da in train_data:
        cell, cl = da
        if torch.cuda.is_available():
            cell = cell.cuda()
            cl = cl.cuda()
        else:
            cell = Variable(cell)
            cl = Variable(cl)

        out = model(cell)
        loss = model.criterion(out, cl)

        print_loss = loss.data.item()
        print_loss = float(print_loss)
        mpt1.append(print_loss)

        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()
    loss = sum(mpt1)
    return model, loss
