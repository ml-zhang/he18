# _*_ coding: utf-8 _*_
'''
@author: Frank Z
@time: 3.8.2019
@target: creating cell classifier use he18_data or other data,
and use the produced model to predict the label of immature cell
three types of cell name cge lge mge were labeled
      __      __
     &&&     &&&              &&
    && $$   $$ &&            &&
   &&   $$ $$   &&          &&
  &&     $$$     &&        &&
 &&      ￥       &&      &&
&&                 &&    &&&&&&&&&&&&

'''
from constant_name import *  # all hyper-para ,easy to change
# from train import train_model_version_two #train_model,bal_train
from test import eval_model, eval_model_version2
from predict import pre_dict
import torch.optim
from data_load import Im_data, DataLoader
import helper_api
import train
import time
import data_modify as dm
import os
def all_3(time_tag, e_stop=1.0):
    ti = time_tag
    # hyper-para
    # path
    torch.manual_seed(15)
    tr_path = r'data/'
    path = r'data/all_3/'
    # load——data
    train_all_l = Im_data(in_path=tr_path + 'f_train_0729_1518.csv', lab_path=tr_path + 'f_test_0729_1518.csv')
    # train_all_l = Im_data(in_path=path+'f_train_24000.csv', lab_path=path+'tr_24000_label.csv')
    # train_all_l = Im_data(in_path=path+'f_train_21000.csv', lab_path=path+'tr_21000_label.csv')
    # train_all_l = Im_data(in_path=path+'f_train_18000.csv', lab_path=path+'tr_18000_label.csv')
    train_all = DataLoader(train_all_l, batch_size=batch_size, shuffle=True)

    test_all_l = Im_data(in_path=path + 'f_test.csv', lab_path=path + 'te_label.csv')
    test_all = DataLoader(test_all_l, batch_size=batch_size, shuffle=False)

    loss_list = []
    acc_list_all = []
    model = None

    for num_ep in range(num_epoch):
        model, loss = train.train_model_version_two(train_all, num_ep, model)
        loss_list.append(loss)
        acc_list = eval_model_version2(model, test_all)
        print('epoch_loss_', num_ep, ':', loss)
        acc_list_all.append(acc_list)
        print(acc_list)
        tag = 1
        for item in acc_list:
            if item < e_stop:
                tag = 0
        if tag:
            break

    helper_api.draw_line(acc_list_all)

    # comd = input('save_model?(yes or exit or no):')
    # comd = 'yes'
    # if comd == 'yes':
    #     torch.save(model.state_dict(), r'model/all_3/A3_' +  ti + '.pkl')
    # elif comd == 'exit':
    #     exit()
    # else:
    #     pass
    # comd = input('use_predict_fun?(yes/no):')
    # comd = 'yes'

    # if comd == 'yes':
    #     pre_batch_size = 100
    #     pre_data_l = Im_data(in_path=pre_path)
    #     pre_data = DataLoader(pre_data_l, batch_size=pre_batch_size, shuffle=False)
    #     pre_dict(model, pre_data, save_name=r'outcome/all_3/' + ti + '.csv')

def run():
    time_tag = time.strftime('%m%d_%H%M', time.localtime(time.time()))
    all_3(time_tag, e_stop=0.95)

def all_3_multi(tr, tr_l, f_tag, e_stop=0.8, bias=0.01):
    path = r'data/all_3/'
    # load——data
    train_all_l = Im_data(in_path=tr, lab_path=tr_l)
    train_all = DataLoader(train_all_l, batch_size=batch_size, shuffle=True)

    test_all_l = Im_data(in_path=path + 'f_test.csv', lab_path=path + 'te_label.csv')
    test_all = DataLoader(test_all_l, batch_size=batch_size, shuffle=False)

    pre_batch_size = 100
    pre_data_l = Im_data(in_path=pre_path)
    pre_data = DataLoader(pre_data_l, batch_size=pre_batch_size, shuffle=False)

    num = []
    for model_tag in range(1, 6):
        time_tag = time.strftime('%H%M%S', time.localtime(time.time()))
        loss_list = []
        acc_list_all = []
        model = None

        for num_ep in range(num_epoch):
            model, loss = train.train_model_version_two(train_all, num_ep, model, model_tag)
            loss_list.append(loss)
            acc_list = eval_model_version2(model, test_all)
            print('epoch_loss_', num_ep, ':', loss)
            acc_list_all.append(acc_list)
            print(acc_list)
            if acc_list[0] - acc_list[1] > bias and acc_list[0] - acc_list[1] > bias:
                tag = 1
                for item in acc_list:
                    if item < e_stop:
                        tag = 0
                if tag:
                    break

        # helper_api.draw_line(acc_list_all)

        # comd = input('save_model?(yes or exit or no):')
        comd = 'yes'
        if comd == 'yes':
            torch.save(model.state_dict(), r'model/all_3_mul/A3_' + time_tag + '.pkl')
        elif comd == 'exit':
            exit()
        else:
            pass
        # comd = input('use_predict_fun?(yes/no):')
        # comd = 'yes'

        if comd == 'yes':
            if not os.path.exists(r'D:\PROJECT\finall_at\outcome\all_3_mul' + '\\' + f_tag):
                os.makedirs(r'D:\PROJECT\finall_at\outcome\all_3_mul' + '\\' + f_tag)
            s_name = r'D:\PROJECT\finall_at\outcome\all_3_mul' + '\\' + f_tag + '\\' + time_tag + '.csv'
            num_cge = pre_dict(model, pre_data, save_name=s_name)
            num.append(num_cge)
    for i in num:
        print(i)

def run_multi(ncge=2500, nlge=2500, nmge=2500, rand_seed=15):
    time_tag = time.strftime('%m%d_%H%M', time.localtime(time.time()))
    tr, tr_l = dm.direct_steam(time_tag, ncge, nlge, nmge, rand_seed)
    print(tr)
    all_3_multi(tr, tr_l, time_tag, e_stop=0.8, bias=0.001)


def run_he18(tr, tr_l, te, te_l, gene, pre_p, e_stop=0.8, bias=0.01):
    f_tag = time.strftime('%m%d_%H%M', time.localtime(time.time()))
    path = 'data/'
    # load——data
    train_all_l = Im_data(in_path=path + gene + tr, lab_path=path + gene + tr_l)
    train_all = DataLoader(train_all_l, batch_size=batch_size, shuffle=True)

    test_all_l = Im_data(in_path=path + gene + te, lab_path=path + gene + te_l)
    test_all = DataLoader(test_all_l, batch_size=batch_size, shuffle=False)

    pre_batch_size = 100
    pre_data_l = Im_data(in_path=path + gene + pre_p)
    pre_data = DataLoader(pre_data_l, batch_size=pre_batch_size, shuffle=False)

    for model_tag in range(1, 6):
        # training five different model with different depth at the same time.
        time_tag = time.strftime('%H%M%S', time.localtime(time.time()))
        loss_list = []
        acc_list_all = []
        model = None

        for num_ep in range(num_epoch):
            model, loss = train.train_model_version_two(train_all, num_ep, model, model_tag)
            loss_list.append(loss)
            acc_list = eval_model_version2(model, test_all)
            print('epoch_loss_', num_ep, ':', loss)
            acc_list_all.append(acc_list)
            print(acc_list)
            if acc_list[0] - acc_list[1] > bias and acc_list[0] - acc_list[1] > bias:
                tag = 1
                for item in acc_list:
                    if item < e_stop:
                        tag = 0
                if tag:
                    break

        helper_api.draw_line(acc_list_all)

        comd = input('save_model?(yes or exit or no):')
        if comd == 'yes':
            torch.save(model.state_dict(), r'model/' + gene + time_tag + '.pkl')
        elif comd == 'exit':
            exit()
        else:
            pass

        comd = input('use_predict_fun?(yes/no):')
        if comd == 'yes':
            if not os.path.exists('outcome/' + gene + f_tag):
                os.makedirs('outcome/' + gene + f_tag)
            s_name = 'outcome/' + gene + f_tag + '/' + time_tag + '.csv'
            _ = pre_dict(model, pre_data, save_name=s_name)
            # num_cge = pre_dict(model, pre_data, save_name=s_name)
    #         num.append(num_cge)
    # for i in num:
    #     print(i)


if __name__ == '__main__':
    torch.manual_seed(15)   #the Rand-seed in main is a global influence factor ，GOOD
    ## run all data
    # run()

    # run he18 with different compressed genes data
    # use i to select data , need to change input dim of first layer by hand if config.py is not used
    fi_list = [['278gene/', '278_gene_tr.csv', '278_gene_tr_l.csv', '278_gene_te.csv', '278_gene_te_l.csv',
                'pre_278_gene.csv'],
               ['1046gene/', '1046_gene_tr.csv', '1046_gene_tr_l.csv', '1046_gene_te.csv', '1046_gene_te_l.csv',
                'pre_1046_gene.csv'],
               ['allgene/', 'all_gene_tr.csv', 'all_gene_tr_l.csv', 'all_gene_te.csv', 'all_gene_te_l.csv',
                'pre_all_gene.csv'],
               ['3000gene/', '3000_gene_tr.csv', '3000_gene_tr_l.csv', '3000_gene_te.csv', '3000_gene_te_l.csv',
                'pre_3000_gene.csv']
               ]
    i = 2   #with all genes
    run_he18(tr=fi_list[i][1], tr_l=fi_list[i][2], te=fi_list[i][3], te_l=fi_list[i][4], gene=fi_list[i][0],
             pre_p=fi_list[i][5], e_stop=1)